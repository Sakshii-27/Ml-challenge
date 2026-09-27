import os
import sys
import collections
import pandas as pd
import numpy as np

def run_eda():
    print("=== STARTING COMPREHENSIVE EDA ===")
    
    # 1. Ground truth analysis
    print("\n--- 1. Ground Truth Analysis ---")
    gt_path = "dataset/train/train_ground_truth.tsv"
    
    total_s1_gt = 0
    singletons = 0
    match_counts = collections.Counter()
    s2_matches_count = 0
    s3_matches_count = 0
    sample_matches = []
    
    with open(gt_path, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split('\t')
        for i, line in enumerate(f):
            total_s1_gt += 1
            parts = line.rstrip('\n').split('\t')
            s1_id = parts[0]
            matches_str = parts[1] if len(parts) > 1 else ""
            if not matches_str.strip():
                singletons += 1
                match_counts[0] += 1
            else:
                matches = matches_str.split(',')
                num_m = len(matches)
                match_counts[num_m] += 1
                for m in matches:
                    if m.startswith('S2-'):
                        s2_matches_count += 1
                    elif m.startswith('S3-'):
                        s3_matches_count += 1
                if len(sample_matches) < 5:
                    sample_matches.append((s1_id, matches))
                    
    print(f"Total Source 1 entities in GT: {total_s1_gt:,}")
    print(f"Singletons (0 matches): {singletons:,} ({singletons / total_s1_gt * 100:.2f}%)")
    print(f"Entities with >= 1 match: {total_s1_gt - singletons:,} ({(total_s1_gt - singletons) / total_s1_gt * 100:.2f}%)")
    print(f"Total S2 matched references: {s2_matches_count:,}")
    print(f"Total S3 matched references: {s3_matches_count:,}")
    print("Match count distribution (matches per S1 entity):")
    for k in sorted(match_counts.keys())[:15]:
        print(f"  {k} matches: {match_counts[k]:,} entities ({match_counts[k] / total_s1_gt * 100:.2f}%)")
    if len(match_counts) > 15:
        more = sum(match_counts[k] for k in match_counts if k >= 15)
        print(f"  >=15 matches: {more:,} entities ({more / total_s1_gt * 100:.2f}%)")

    # 2. Country distributions across train and test
    print("\n--- 2. Country Distributions ---")
    files_to_check = [
        ("Train S1", "dataset/train/train_source1.tsv"),
        ("Train S2", "dataset/train/train_source2.tsv"),
        ("Train S3", "dataset/train/train_source3.tsv"),
        ("Test S1", "dataset/test/test_source1.tsv"),
        ("Test S2", "dataset/test/test_source2.tsv"),
        ("Test S3", "dataset/test/test_source3.tsv"),
    ]
    
    for label, path in files_to_check:
        country_counts = collections.Counter()
        missing_name = 0
        missing_address = 0
        total_rows = 0
        with open(path, 'r', encoding='utf-8') as f:
            header = f.readline().strip().split('\t')
            for line in f:
                total_rows += 1
                parts = line.rstrip('\n').split('\t')
                # entity_id, business_name, business_address, country
                b_name = parts[1] if len(parts) > 1 else ""
                b_addr = parts[2] if len(parts) > 2 else ""
                c = parts[3] if len(parts) > 3 else "MISSING"
                if not b_name.strip(): missing_name += 1
                if not b_addr.strip(): missing_address += 1
                country_counts[c] += 1
        print(f"{label} ({total_rows:,} rows):")
        for c, cnt in country_counts.most_common():
            print(f"  {c}: {cnt:,} ({cnt/total_rows*100:.2f}%)")
        print(f"  Missing names: {missing_name:,}, Missing addresses: {missing_address:,}")

    # 3. Matching properties: Do entities match across different countries?
    print("\n--- 3. Cross-Country Match Consistency Check (Sample 100k) ---")
    # Load 100k S1 country mapping
    s1_country = {}
    with open("dataset/train/train_source1.tsv", 'r', encoding='utf-8') as f:
        f.readline()
        for i, line in enumerate(f):
            parts = line.rstrip('\n').split('\t')
            s1_country[parts[0]] = parts[3] if len(parts) > 3 else ""
            if i >= 100000:
                break
                
    s2_country = {}
    with open("dataset/train/train_source2.tsv", 'r', encoding='utf-8') as f:
        f.readline()
        for i, line in enumerate(f):
            parts = line.rstrip('\n').split('\t')
            s2_country[parts[0]] = parts[3] if len(parts) > 3 else ""
            if i >= 300000:
                break
                
    s3_country = {}
    with open("dataset/train/train_source3.tsv", 'r', encoding='utf-8') as f:
        f.readline()
        for i, line in enumerate(f):
            parts = line.rstrip('\n').split('\t')
            s3_country[parts[0]] = parts[3] if len(parts) > 3 else ""
            if i >= 300000:
                break

    same_country = 0
    diff_country = 0
    checked = 0
    with open(gt_path, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            parts = line.rstrip('\n').split('\t')
            s1_id = parts[0]
            if s1_id in s1_country:
                c1 = s1_country[s1_id]
                matches = parts[1].split(',') if len(parts) > 1 and parts[1].strip() else []
                for m in matches:
                    c_other = s2_country.get(m) or s3_country.get(m)
                    if c_other:
                        checked += 1
                        if c_other == c1:
                            same_country += 1
                        else:
                            diff_country += 1
    print(f"Evaluated matches where both IDs are mapped: {checked:,}")
    print(f"Same country matches: {same_country:,} ({(same_country/checked*100) if checked else 0:.2f}%)")
    print(f"Different country matches: {diff_country:,} ({(diff_country/checked*100) if checked else 0:.2f}%)")

if __name__ == "__main__":
    run_eda()
