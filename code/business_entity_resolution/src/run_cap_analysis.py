"""
Tests recall ceiling without any top_k cap (all candidates sharing >= 1 key)
versus different top_k caps (50, 100, 200, 500, no cap)
to see the exact trade-off curve between candidate set size and recall loss.
"""
import sys
import csv
import time
import collections
from blocking_harness import create_val_split, load_ground_truth, evaluate_blocking, print_evaluation_report
from blocking_keys import extract_name_keys, extract_address_keys

def run_cap_analysis(val_size=5000, max_cands_per_key=1000):
    val_s1_ids = create_val_split(val_s1_size=val_size)
    ground_truth = load_ground_truth(val_s1_ids)
    
    print(f"Loading S1 validation records ({len(val_s1_ids):,} entities)...")
    s1_records = {}
    s1_countries = {}
    with open("dataset/train/train_source1.tsv", "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            s1_id = row[0]
            if s1_id in val_s1_ids:
                s1_records[s1_id] = {
                    "name": row[1],
                    "address": row[2],
                    "country": row[3]
                }
                s1_countries[s1_id] = row[3]
                
    inverted_index = {
        "US": collections.defaultdict(list),
        "India": collections.defaultdict(list)
    }
    
    cand_count = 0
    t0 = time.time()
    for filename in ["dataset/train/train_source2.tsv", "dataset/train/train_source3.tsv"]:
        print(f"  Indexing {filename}...")
        with open(filename, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)
            for row in reader:
                cand_id = row[0]
                name = row[1]
                address = row[2]
                country = row[3]
                if country not in inverted_index:
                    continue
                cand_count += 1
                n_keys = extract_name_keys(name)
                a_keys = extract_address_keys(address)
                c_idx = inverted_index[country]
                for k in set(n_keys + a_keys):
                    if len(c_idx[k]) < max_cands_per_key:
                        c_idx[k].append(cand_id)
                        
    print(f"Indexed {cand_count:,} records in {time.time() - t0:.1f}s")
    
    # Precompute candidate hits for all S1 entities
    print("Scoring candidates for S1 entities...")
    s1_ranked_candidates = {}
    for s1_id, rec in s1_records.items():
        c_idx = inverted_index.get(rec["country"])
        if not c_idx:
            continue
        all_keys = set(extract_name_keys(rec["name"]) + extract_address_keys(rec["address"]))
        cand_hits = collections.Counter()
        for k in all_keys:
            postings = c_idx.get(k)
            if postings and len(postings) < max_cands_per_key:
                for cid in postings:
                    cand_hits[cid] += 1
        # Store sorted by score descending
        s1_ranked_candidates[s1_id] = cand_hits.most_common()

    # Now evaluate across caps: 50, 100, 200, 500, and No Cap (Infinity)
    caps = [30, 50, 100, 200, 500, None]
    print("\n" + "="*80)
    print("                 CAP VS RECALL CEILING SENSITIVITY TABLE")
    print("="*80)
    print(f"{'Cap (Top-K)':<12} | {'Pair Recall':<12} | {'Macro Recall':<14} | {'Mean Cands':<12} | {'Med Cands':<12} | {'P90 Cands':<10}")
    print("-"*80)
    
    for cap in caps:
        cand_dict = {}
        for s1_id, ranked in s1_ranked_candidates.items():
            if cap is None:
                cand_dict[s1_id] = set(cid for cid, _ in ranked)
            else:
                cand_dict[s1_id] = set(cid for cid, _ in ranked[:cap])
                
        rep = evaluate_blocking(cand_dict, ground_truth, s1_countries)
        cap_str = str(cap) if cap is not None else "NO CAP (ALL)"
        print(f"{cap_str:<12} | {rep['overall_pair_recall_ceiling']*100:<11.2f}% | {rep['macro_entity_recall_ceiling']*100:<13.2f}% | {rep['candidates_mean']:<12.1f} | {rep['candidates_median']:<12.1f} | {rep['candidates_p90']:<10.1f}")
    print("="*80 + "\n")

if __name__ == "__main__":
    run_cap_analysis(val_size=5000)
