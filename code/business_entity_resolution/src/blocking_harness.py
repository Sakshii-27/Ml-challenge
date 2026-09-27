"""
Harness to build a stratified local validation split and measure:
1. True match recall ceiling
2. Candidate reduction ratio and candidate distribution (mean, median, p90, max per S1 entity)
3. Breakdown across countries and sources
"""
import os
import csv
import random
import collections
import numpy as np

def create_val_split(val_s1_size=20000, seed=42, split_file="val_split_s1_ids.txt"):
    """
    Selects a stratified sample of S1 entity IDs by country (US, India).
    """
    if os.path.exists(split_file):
        with open(split_file, "r") as f:
            val_ids = set(line.strip() for line in f if line.strip())
        print(f"Loaded existing validation split of {len(val_ids):,} S1 IDs from {split_file}")
        return val_ids

    print(f"Creating reproducible stratified validation split of {val_s1_size:,} S1 entities...")
    random.seed(seed)
    
    us_ids = []
    in_ids = []
    
    with open("dataset/train/train_source1.tsv", "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            s1_id = row[0]
            country = row[3]
            if country == "US":
                us_ids.append(s1_id)
            elif country == "India":
                in_ids.append(s1_id)
                
    # 60% US, 40% India to match population distribution
    n_us = int(val_s1_size * 0.6)
    n_in = val_s1_size - n_us
    
    val_us = random.sample(us_ids, n_us)
    val_in = random.sample(in_ids, n_in)
    
    val_ids = set(val_us + val_in)
    with open(split_file, "w") as f:
        for vid in sorted(val_ids):
            f.write(f"{vid}\n")
            
    print(f"Created validation split: {len(val_us):,} US + {len(val_in):,} India = {len(val_ids):,} S1 IDs saved to {split_file}")
    return val_ids

def load_ground_truth(val_s1_ids):
    """
    Loads true matches for S1 entities in val_s1_ids.
    Returns: dict {s1_id: set(matched_s2_or_s3_ids)}
    """
    gt = {}
    with open("dataset/train/train_ground_truth.tsv", "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            s1_id = row[0]
            if s1_id in val_s1_ids:
                matches = set(row[1].split(",")) if len(row) > 1 and row[1].strip() else set()
                gt[s1_id] = matches
    return gt

def evaluate_blocking(candidates_dict, ground_truth_dict, s1_country_map=None):
    """
    Evaluates blocking candidates against ground truth.
    candidates_dict: {s1_id: set(candidate_ids)}
    ground_truth_dict: {s1_id: set(true_match_ids)}
    """
    total_s1 = len(ground_truth_dict)
    s1_with_matches = 0
    total_true_matches = 0
    captured_true_matches = 0
    
    # Per-entity recall
    entity_recalls = []
    candidate_counts = []
    
    country_stats = collections.defaultdict(lambda: {
        "total_true": 0,
        "captured_true": 0,
        "s1_count": 0,
        "recalls": [],
        "cand_counts": []
    })
    
    for s1_id, true_matches in ground_truth_dict.items():
        cands = candidates_dict.get(s1_id, set())
        n_cands = len(cands)
        candidate_counts.append(n_cands)
        
        country = s1_country_map.get(s1_id, "Unknown") if s1_country_map else "All"
        c_stat = country_stats[country]
        c_stat["s1_count"] += 1
        c_stat["cand_counts"].append(n_cands)
        
        n_true = len(true_matches)
        if n_true > 0:
            s1_with_matches += 1
            total_true_matches += n_true
            captured = len(true_matches.intersection(cands))
            captured_true_matches += captured
            rec = captured / n_true
            entity_recalls.append(rec)
            
            c_stat["total_true"] += n_true
            c_stat["captured_true"] += captured
            c_stat["recalls"].append(rec)
        else:
            # Singleton: 100% recall trivially since there are no links to find
            entity_recalls.append(1.0)
            c_stat["recalls"].append(1.0)

    overall_pair_recall = (captured_true_matches / total_true_matches) if total_true_matches > 0 else 1.0
    macro_recall = float(np.mean(entity_recalls)) if entity_recalls else 1.0
    
    cand_counts_arr = np.array(candidate_counts)
    
    report = {
        "total_s1": total_s1,
        "s1_with_matches": s1_with_matches,
        "singletons": total_s1 - s1_with_matches,
        "total_true_links": total_true_matches,
        "captured_true_links": captured_true_matches,
        "overall_pair_recall_ceiling": overall_pair_recall,
        "macro_entity_recall_ceiling": macro_recall,
        "candidates_mean": float(np.mean(cand_counts_arr)),
        "candidates_median": float(np.median(cand_counts_arr)),
        "candidates_p90": float(np.percentile(cand_counts_arr, 90)),
        "candidates_p99": float(np.percentile(cand_counts_arr, 99)),
        "candidates_max": int(np.max(cand_counts_arr)) if len(cand_counts_arr) > 0 else 0,
        "country_breakdown": {}
    }
    
    for c, stat in country_stats.items():
        c_tot = stat["total_true"]
        c_cap = stat["captured_true"]
        c_arr = np.array(stat["cand_counts"])
        report["country_breakdown"][c] = {
            "s1_count": stat["s1_count"],
            "pair_recall": (c_cap / c_tot) if c_tot > 0 else 1.0,
            "macro_recall": float(np.mean(stat["recalls"])),
            "cand_mean": float(np.mean(c_arr)),
            "cand_median": float(np.median(c_arr)),
            "cand_max": int(np.max(c_arr))
        }
        
    return report

def print_evaluation_report(report):
    print("\n=======================================================")
    print("           BLOCKING RECALL CEILING REPORT              ")
    print("=======================================================")
    print(f"Total S1 entities evaluated:    {report['total_s1']:,}")
    print(f"Entities with true matches:     {report['s1_with_matches']:,}")
    print(f"Singletons (0 matches):         {report['singletons']:,}")
    print(f"Total true match pairs:         {report['total_true_links']:,}")
    print(f"Captured true match pairs:      {report['captured_true_links']:,}")
    print("-------------------------------------------------------")
    print(f">> PAIR RECALL CEILING:         {report['overall_pair_recall_ceiling']*100:.2f}%")
    print(f">> MACRO S1 RECALL CEILING:     {report['macro_entity_recall_ceiling']*100:.2f}%")
    print("-------------------------------------------------------")
    print("Candidate Count per S1 Distribution:")
    print(f"  Mean:   {report['candidates_mean']:.1f}")
    print(f"  Median: {report['candidates_median']:.1f}")
    print(f"  P90:    {report['candidates_p90']:.1f}")
    print(f"  P99:    {report['candidates_p99']:.1f}")
    print(f"  Max:    {report['candidates_max']:,}")
    print("-------------------------------------------------------")
    print("Breakdown by Country:")
    for country, stat in report["country_breakdown"].items():
        print(f"  [{country}] S1 count: {stat['s1_count']:,} | Pair Recall: {stat['pair_recall']*100:.2f}% | Macro Recall: {stat['macro_recall']*100:.2f}% | Mean Cands: {stat['cand_mean']:.1f} | Med Cands: {stat['cand_median']:.1f}")
    print("=======================================================\n")
