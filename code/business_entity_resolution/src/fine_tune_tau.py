"""
Fine-grained threshold optimization extending above 0.90 (0.85 to 0.99)
to pinpoint the exact peak of Macro F_0.5.
"""
import sys
import csv
import collections
import numpy as np
import lightgbm as lgb
from blocking_harness import create_val_split, load_ground_truth
from blocking_keys import extract_name_keys, extract_address_keys
from features import extract_pairwise_features
from metric import compute_macro_f05

def fine_tune():
    val_s1_ids = create_val_split(val_s1_size=5000, seed=42, split_file="val_split_s1_ids.txt")
    ground_truth = load_ground_truth(val_s1_ids)
    
    # Load model
    model = lgb.Booster(model_file="lightgbm_matching_model.txt")
    
    # Load S1 validation info
    s1_all_info = {}
    with open("dataset/train/train_source1.tsv", "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if row[0] in val_s1_ids:
                s1_all_info[row[0]] = (row[1], row[2], row[3])
                
    val_countries = {s1_id: s1_all_info[s1_id][2] for s1_id in val_s1_ids}
    
    # Index S2 and S3 for validation
    inverted_index = {"US": collections.defaultdict(list), "India": collections.defaultdict(list)}
    cand_records = {}
    for filename in ["dataset/train/train_source2.tsv", "dataset/train/train_source3.tsv"]:
        with open(filename, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)
            for row in reader:
                cand_id, name, addr, country = row[0], row[1], row[2], row[3]
                if country in inverted_index:
                    cand_records[cand_id] = (name, addr, country)
                    n_keys = extract_name_keys(name)
                    a_keys = extract_address_keys(addr)
                    for k in set(n_keys + a_keys):
                        if len(inverted_index[country][k]) < 1000:
                            inverted_index[country][k].append(cand_id)
                            
    # Candidate retrieval and prediction
    val_feats = []
    val_meta = []
    for s1_id in val_s1_ids:
        s1_name, s1_addr, s1_country = s1_all_info[s1_id]
        c_idx = inverted_index.get(s1_country)
        if not c_idx:
            continue
        all_keys = set(extract_name_keys(s1_name) + extract_address_keys(s1_addr))
        cand_hits = collections.Counter()
        for k in all_keys:
            postings = c_idx.get(k)
            if postings and len(postings) < 1000:
                for cid in postings:
                    cand_hits[cid] += 1
        ranked = cand_hits.most_common(200)
        for cid, shared_k in ranked:
            c_name, c_addr, c_country = cand_records[cid]
            feat = extract_pairwise_features(s1_name, s1_addr, s1_country, cid, c_name, c_addr, c_country, shared_k)
            val_feats.append(feat)
            val_meta.append((s1_id, cid))
            
    val_probs = model.predict(np.array(val_feats, dtype=np.float32))
    s1_pairs = collections.defaultdict(list)
    for (s1_id, cid), p in zip(val_meta, val_probs):
        s1_pairs[s1_id].append((cid, p))
        
    print("\n" + "="*80)
    print("             HIGH-PRECISION THRESHOLD TUNING (0.85 -> 0.99)")
    print("="*80)
    
    high_thresholds = [0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    for tau in high_thresholds:
        preds = {}
        for s1_id in ground_truth:
            preds[s1_id] = {cid for cid, p in s1_pairs.get(s1_id, []) if p >= tau}
        overall, c_scores = compute_macro_f05(preds, ground_truth, val_countries)
        print(f"tau={tau:.2f} -> Macro F_0.5 = {overall*100:.3f}% | US: {c_scores.get('US',0)*100:.3f}% | India: {c_scores.get('India',0)*100:.3f}%")
    print("="*80 + "\n")

if __name__ == "__main__":
    fine_tune()
