"""
End-to-End Training and Threshold Tuning for LightGBM Matching Classifier.
Handles:
1. Negative subsampling (1:8 positive to negative ratio) to handle extreme candidate imbalance.
2. Direct feature extraction on pairs using rapidfuzz and normalized string signals.
3. Training LightGBM binary classifier.
4. Validation inference on full Top-200 candidate sets.
5. Grid search for optimal global tau and per-country tau (US vs India) maximizing macro F_0.5.
"""
import sys
import csv
import time
import random
import collections
import numpy as np
import lightgbm as lgb
from blocking_harness import create_val_split, load_ground_truth
from blocking_keys import extract_name_keys, extract_address_keys
from features import extract_pairwise_features, FEATURE_NAMES
from metric import compute_macro_f05, optimize_thresholds

def run_train_and_tune(n_train_s1=6000, n_val_s1=3000, neg_subsample_ratio=8, top_k_cap=200):
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)
    
    # 1. Select disjoint Train and Val S1 entities
    print("Step 1: Selecting S1 entities for Training and Validation...")
    val_s1_ids = create_val_split(val_s1_size=n_val_s1, seed=42, split_file="val_split_s1_ids.txt")
    
    train_s1_ids = set()
    s1_all_info = {}
    with open("dataset/train/train_source1.tsv", "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            s1_id = row[0]
            if s1_id in val_s1_ids:
                s1_all_info[s1_id] = (row[1], row[2], row[3])
            elif len(train_s1_ids) < n_train_s1:
                train_s1_ids.add(s1_id)
                s1_all_info[s1_id] = (row[1], row[2], row[3])
                
    print(f"Selected {len(train_s1_ids):,} Train S1 entities and {len(val_s1_ids):,} Validation S1 entities.")
    
    # Load ground truth for all selected S1
    ground_truth_all = {}
    with open("dataset/train/train_ground_truth.tsv", "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            s1_id = row[0]
            if s1_id in s1_all_info:
                ms = set(row[1].split(",")) if len(row) > 1 and row[1].strip() else set()
                ground_truth_all[s1_id] = ms
                
    val_gt = {s1_id: ground_truth_all[s1_id] for s1_id in val_s1_ids}
    val_countries = {s1_id: s1_all_info[s1_id][2] for s1_id in val_s1_ids}
    
    # 2. Build inverted index on S2 & S3
    print("\nStep 2: Building Country Inverted Index from S2 & S3...")
    inverted_index = {
        "US": collections.defaultdict(list),
        "India": collections.defaultdict(list)
    }
    
    needed_cands = set()
    # We also need candidate records in memory for quick feature extraction
    cand_records = {}
    
    t0 = time.time()
    total_indexed = 0
    for filename in ["dataset/train/train_source2.tsv", "dataset/train/train_source3.tsv"]:
        print(f"  Streaming {filename}...")
        with open(filename, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)
            for row in reader:
                total_indexed += 1
                cand_id, name, addr, country = row[0], row[1], row[2], row[3]
                if country not in inverted_index:
                    continue
                # Store record for fast feature generation
                cand_records[cand_id] = (name, addr, country)
                n_keys = extract_name_keys(name)
                a_keys = extract_address_keys(addr)
                c_idx = inverted_index[country]
                for k in set(n_keys + a_keys):
                    if len(c_idx[k]) < 1000:
                        c_idx[k].append(cand_id)
                        
    print(f"  Indexed {total_indexed:,} records in {time.time() - t0:.1f}s")
    
    # 3. Generate Training Pairs with Negative Subsampling
    print("\nStep 3: Generating Training pairs (subsampling negatives 1:{})...".format(neg_subsample_ratio))
    X_train = []
    y_train = []
    
    pos_count = 0
    neg_count = 0
    
    for s1_id in train_s1_ids:
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
                    
        # Apply Top-K cap
        ranked = cand_hits.most_common(top_k_cap)
        true_matches = ground_truth_all.get(s1_id, set())
        
        pos_cands = []
        neg_cands = []
        
        for cid, shared_k in ranked:
            if cid in true_matches:
                pos_cands.append((cid, shared_k))
            else:
                neg_cands.append((cid, shared_k))
                
        # Also ensure true matches are included even if ranked lower (positives are precious)
        for tm in true_matches:
            if tm in cand_records and tm not in dict(pos_cands):
                pos_cands.append((tm, cand_hits.get(tm, 1)))
                
        # Negative subsampling: keep at most neg_subsample_ratio * len(pos_cands)
        n_pos = len(pos_cands)
        if n_pos > 0:
            max_negs = max(n_pos * neg_subsample_ratio, 5)
            selected_negs = random.sample(neg_cands, min(len(neg_cands), max_negs))
        else:
            # For singletons, keep a small sample of hard negatives
            selected_negs = random.sample(neg_cands, min(len(neg_cands), 4))
            
        for cid, shared_k in pos_cands:
            c_name, c_addr, c_country = cand_records[cid]
            feat = extract_pairwise_features(s1_name, s1_addr, s1_country, cid, c_name, c_addr, c_country, shared_k)
            X_train.append(feat)
            y_train.append(1)
            pos_count += 1
            
        for cid, shared_k in selected_negs:
            c_name, c_addr, c_country = cand_records[cid]
            feat = extract_pairwise_features(s1_name, s1_addr, s1_country, cid, c_name, c_addr, c_country, shared_k)
            X_train.append(feat)
            y_train.append(0)
            neg_count += 1

    print(f"Training dataset constructed: {len(y_train):,} pairs ({pos_count:,} positive, {neg_count:,} negative). Ratio 1:{neg_count/pos_count:.1f}")
    
    # 4. Train LightGBM model
    print("\nStep 4: Training LightGBM Matching Model...")
    X_train_mat = np.array(X_train, dtype=np.float32)
    y_train_vec = np.array(y_train, dtype=np.int32)
    
    train_data = lgb.Dataset(X_train_mat, label=y_train_vec, feature_name=FEATURE_NAMES)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.08,
        'num_leaves': 31,
        'max_depth': 6,
        'min_data_in_leaf': 20,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 1,
        'verbose': -1,
        'n_jobs': 4,
        'seed': 42
    }
    
    model = lgb.train(params, train_data, num_boost_round=150)
    print("LightGBM model training finished.")
    
    # Feature importances
    importances = model.feature_importance(importance_type='gain')
    feat_imp = sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True)
    print("\nTop 10 Most Important Features (by gain):")
    for fname, gain in feat_imp[:10]:
        print(f"  {fname:<20}: {gain:.1f}")

    # 5. Validation Inference (Top-200 candidates per S1)
    print("\nStep 5: Performing Validation Inference on Top-{} candidates...".format(top_k_cap))
    val_pairs = []
    val_feats = []
    val_meta = [] # (s1_id, cid)
    
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
                    
        # Strict Top-200 Cap
        ranked = cand_hits.most_common(top_k_cap)
        for cid, shared_k in ranked:
            c_name, c_addr, c_country = cand_records[cid]
            feat = extract_pairwise_features(s1_name, s1_addr, s1_country, cid, c_name, c_addr, c_country, shared_k)
            val_feats.append(feat)
            val_meta.append((s1_id, cid))
            
    print(f"Extracting predictions on {len(val_feats):,} validation candidate pairs...")
    val_feats_mat = np.array(val_feats, dtype=np.float32)
    val_probs = model.predict(val_feats_mat)
    
    for (s1_id, cid), prob in zip(val_meta, val_probs):
        val_pairs.append((s1_id, cid, float(prob)))

    # 6. Optimize Thresholds (Global & Per-Country)
    print("\nStep 6: Optimizing Threshold tau for Macro F_0.5...")
    best_global_tau, best_country_taus = optimize_thresholds(val_pairs, val_gt, val_countries)
    
    print("\n" + "="*80)
    print("                    FINAL MODEL EVALUATION SUMMARY")
    print("="*80)
    print(f"Optimal Global Threshold: tau = {best_global_tau:.2f}")
    for country, (tau_c, score_c) in best_country_taus.items():
        print(f"Optimal {country} Threshold: tau = {tau_c:.2f} (Macro F_0.5 = {score_c*100:.3f}%)")
    print(f"Total pipeline elapsed time: {time.time() - start_time:.1f}s")
    print("="*80 + "\n")
    
    # Save model artifact
    model.save_model("lightgbm_matching_model.txt")
    print("Saved trained model to lightgbm_matching_model.txt")

if __name__ == "__main__":
    run_train_and_tune()
