"""
Test Set Inference Pipeline.
Generates output/matching_results.tsv and output/candidate_pairs.tsv 
for the ML Challenge 2026.
"""
import os
import csv
import time
import collections
import numpy as np
import lightgbm as lgb
from blocking_keys import extract_name_keys, extract_address_keys
from features import extract_pairwise_features

def run_inference(top_k_cap=200, threshold=0.90):
    start_time = time.time()
    os.makedirs("output", exist_ok=True)
    
    print("Loading trained LightGBM model...")
    model = lgb.Booster(model_file="lightgbm_matching_model.txt")
    
    print("\nStep 1: Building Test Set Country Inverted Index from S2 & S3...")
    inverted_index = {
        "US": collections.defaultdict(list),
        "India": collections.defaultdict(list),
        "France": collections.defaultdict(list)
    }
    cand_records = {}
    
    t0 = time.time()
    total_indexed = 0
    for filename in ["dataset/test/test_source2.tsv", "dataset/test/test_source3.tsv"]:
        print(f"  Streaming {filename}...")
        with open(filename, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader) # skip header
            for row in reader:
                total_indexed += 1
                cand_id, name, addr, country = row[0], row[1], row[2], row[3]
                if country not in inverted_index:
                    # Fallback if weird country appears
                    inverted_index[country] = collections.defaultdict(list)
                    
                cand_records[cand_id] = (name, addr, country)
                n_keys = extract_name_keys(name)
                a_keys = extract_address_keys(addr)
                c_idx = inverted_index[country]
                for k in set(n_keys + a_keys):
                    if len(c_idx[k]) < 1000:
                        c_idx[k].append(cand_id)
                        
    print(f"  Indexed {total_indexed:,} test records in {time.time() - t0:.1f}s")
    
    print("\nStep 2: Retrieving candidates and predicting matches for Test S1 entities...")
    
    out_matching = open("output/matching_results.tsv", "w", encoding="utf-8", newline="")
    out_candidates = open("output/candidate_pairs.tsv", "w", encoding="utf-8", newline="")
    
    m_writer = csv.writer(out_matching, delimiter="\t")
    c_writer = csv.writer(out_candidates, delimiter="\t")
    
    m_writer.writerow(["source1_entity_id", "matched_entity_ids"])
    c_writer.writerow(["source1_entity_id", "candidate_entity_ids"])
    
    s1_processed = 0
    t1 = time.time()
    
    # Process S1 sequentially to avoid massive memory footprint for pairs
    with open("dataset/test/test_source1.tsv", "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        
        batch_s1_ids = []
        batch_feats = []
        batch_meta = [] # (s1_id, cid)
        batch_all_cands = collections.defaultdict(list)
        
        for row in reader:
            s1_id, s1_name, s1_addr, s1_country = row[0], row[1], row[2], row[3]
            s1_processed += 1
            
            c_idx = inverted_index.get(s1_country, {})
            all_keys = set(extract_name_keys(s1_name) + extract_address_keys(s1_addr))
            
            cand_hits = collections.Counter()
            for k in all_keys:
                postings = c_idx.get(k)
                if postings and len(postings) < 1000:
                    for cid in postings:
                        cand_hits[cid] += 1
                        
            ranked = cand_hits.most_common(top_k_cap)
            
            batch_s1_ids.append(s1_id)
            for cid, shared_k in ranked:
                c_name, c_addr, c_country = cand_records[cid]
                feat = extract_pairwise_features(s1_name, s1_addr, s1_country, cid, c_name, c_addr, c_country, shared_k)
                batch_feats.append(feat)
                batch_meta.append((s1_id, cid))
                batch_all_cands[s1_id].append(cid)
                
            # Process in batches of 10,000 S1 entities
            if s1_processed % 10000 == 0:
                if batch_feats:
                    probs = model.predict(np.array(batch_feats, dtype=np.float32))
                    s1_matches = collections.defaultdict(list)
                    for (b_s1, b_cid), p in zip(batch_meta, probs):
                        if p >= threshold:
                            s1_matches[b_s1].append(b_cid)
                            
                    for b_s1 in batch_s1_ids:
                        cands = batch_all_cands[b_s1]
                        matches = s1_matches.get(b_s1, [])
                        
                        m_writer.writerow([b_s1, ",".join(matches)])
                        c_writer.writerow([b_s1, ",".join(cands)])
                else:
                    # Empty batch (no candidates found for any S1)
                    for b_s1 in batch_s1_ids:
                        m_writer.writerow([b_s1, ""])
                        c_writer.writerow([b_s1, ""])
                        
                batch_s1_ids = []
                batch_feats = []
                batch_meta = []
                batch_all_cands.clear()
                print(f"  Processed {s1_processed:,} S1 entities... (Time: {time.time()-t1:.1f}s)")
                
        # Final batch
        if batch_s1_ids:
            if batch_feats:
                probs = model.predict(np.array(batch_feats, dtype=np.float32))
                s1_matches = collections.defaultdict(list)
                for (b_s1, b_cid), p in zip(batch_meta, probs):
                    if p >= threshold:
                        s1_matches[b_s1].append(b_cid)
                        
                for b_s1 in batch_s1_ids:
                    cands = batch_all_cands[b_s1]
                    matches = s1_matches.get(b_s1, [])
                    m_writer.writerow([b_s1, ",".join(matches)])
                    c_writer.writerow([b_s1, ",".join(cands)])
            else:
                for b_s1 in batch_s1_ids:
                    m_writer.writerow([b_s1, ""])
                    c_writer.writerow([b_s1, ""])
                    
    out_matching.close()
    out_candidates.close()
    print(f"\nCompleted processing {s1_processed:,} Test S1 entities.")
    print(f"Results saved to output/matching_results.tsv and output/candidate_pairs.tsv")
    print(f"Total Inference Time: {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    run_inference(top_k_cap=200, threshold=0.90)
