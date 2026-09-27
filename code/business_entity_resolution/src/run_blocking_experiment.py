"""
Executes the blocking evaluation on the validation split.
Measures:
1. Candidate count per S1 entity
2. Exact recall ceiling against ground truth
3. Country-wise breakdown (US vs India)
4. Key efficiency & selectivity analysis
"""
import sys
import csv
import time
import collections
from blocking_harness import create_val_split, load_ground_truth, evaluate_blocking, print_evaluation_report
from blocking_keys import extract_name_keys, extract_address_keys

def run_blocking_experiment(val_size=10000, max_cands_per_key=1000, top_k_per_s1=100):
    start_time = time.time()
    val_s1_ids = create_val_split(val_s1_size=val_size)
    ground_truth = load_ground_truth(val_s1_ids)
    
    print(f"\nStep 1: Loading S1 records for validation ({len(val_s1_ids):,} entities)...")
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
                
    # Build inverted index for S2 and S3 partitioned by country
    print("\nStep 2: Building Country-Partitioned Inverted Index from S2 & S3...")
    
    # Inverted index structure: index[country][key] -> list of candidate_ids
    # To manage memory and avoid super-dense stopwords blowup, we track counts
    inverted_index = {
        "US": collections.defaultdict(list),
        "India": collections.defaultdict(list)
    }
    
    # Track candidate metadata
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
                
                # Channel 1: Name keys
                n_keys = extract_name_keys(name)
                # Channel 2: Address keys
                a_keys = extract_address_keys(address)
                
                c_idx = inverted_index[country]
                for k in set(n_keys + a_keys):
                    # Guard against excessive posting list length during build
                    if len(c_idx[k]) < max_cands_per_key:
                        c_idx[k].append(cand_id)
                        
                if cand_count % 2000000 == 0:
                    print(f"    Indexed {cand_count:,} records in {time.time() - t0:.1f}s...")
                    
    print(f"  Total S2+S3 records indexed: {cand_count:,} in {time.time() - t0:.1f}s")
    
    # Step 3: Candidate retrieval for validation S1 entities
    print("\nStep 3: Querying candidates for validation S1 entities...")
    t1 = time.time()
    candidates_dict = collections.defaultdict(set)
    
    for s1_id, rec in s1_records.items():
        country = rec["country"]
        c_idx = inverted_index.get(country)
        if not c_idx:
            continue
            
        n_keys = extract_name_keys(rec["name"])
        a_keys = extract_address_keys(rec["address"])
        all_keys = set(n_keys + a_keys)
        
        # Score candidates by frequency of shared keys (overlap score)
        cand_key_hits = collections.Counter()
        for k in all_keys:
            postings = c_idx.get(k)
            if postings and len(postings) < max_cands_per_key:
                for cid in postings:
                    cand_key_hits[cid] += 1
                    
        # Select top-k candidates for this S1 entity
        top_cands = [cid for cid, count in cand_key_hits.most_common(top_k_per_s1)]
        candidates_dict[s1_id] = set(top_cands)
        
    print(f"  Candidate query completed in {time.time() - t1:.1f}s")
    
    # Step 4: Evaluate with harness
    print("\nStep 4: Scoring blocking results with Ground Truth...")
    report = evaluate_blocking(candidates_dict, ground_truth, s1_countries)
    print_evaluation_report(report)
    print(f"Total experiment time: {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    val_sz = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    run_blocking_experiment(val_size=val_sz)
