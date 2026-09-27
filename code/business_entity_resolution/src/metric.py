"""
Macro F_0.5 Evaluation and Threshold Optimizer.
Supports global threshold optimization and per-country threshold optimization.
Evaluates singletons correctly (empty list = 1.0, false positive = 0.0).
"""
import numpy as np
import collections

def compute_macro_f05(predictions, ground_truth, s1_countries=None):
    """
    Computes macro F_0.5 score across all S1 entities.
    predictions: dict {s1_id: set(predicted_candidate_ids)}
    ground_truth: dict {s1_id: set(true_candidate_ids)}
    s1_countries: optional dict {s1_id: country_str}
    
    Returns:
      overall_f05, country_f05_dict
    """
    entity_scores = []
    country_scores = collections.defaultdict(list)
    
    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        country = s1_countries.get(s1_id, "Unknown") if s1_countries else "All"
        
        n_true = len(true_set)
        n_pred = len(pred_set)
        
        if n_true == 0:
            # Singleton handling
            score = 1.0 if n_pred == 0 else 0.0
        elif n_pred == 0:
            score = 0.0
        else:
            tp = len(true_set.intersection(pred_set))
            if tp == 0:
                score = 0.0
            else:
                prec = tp / n_pred
                rec = tp / n_true
                # F_0.5 formula: (1.25 * P * R) / (0.25 * P + R)
                denom = 0.25 * prec + rec
                score = (1.25 * prec * rec) / denom if denom > 0 else 0.0
                
        entity_scores.append(score)
        country_scores[country].append(score)
        
    overall_macro_f05 = float(np.mean(entity_scores)) if entity_scores else 0.0
    country_f05_summary = {c: float(np.mean(scores)) for c, scores in country_scores.items()}
    
    return overall_macro_f05, country_f05_summary

def optimize_thresholds(val_pairs, ground_truth, s1_countries):
    """
    Searches for optimal thresholds (global and per-country) on validation pairs.
    val_pairs: list of tuples (s1_id, cand_id, pred_probability)
    ground_truth: dict {s1_id: set(true_match_ids)}
    s1_countries: dict {s1_id: country_str}
    """
    # Group pairs by S1
    s1_pairs = collections.defaultdict(list)
    for s1_id, cand_id, prob in val_pairs:
        s1_pairs[s1_id].append((cand_id, prob))
        
    # Test grid of thresholds from 0.10 to 0.90
    threshold_grid = np.linspace(0.1, 0.9, 17)
    
    print("\n--- GLOBAL THRESHOLD TUNING ---")
    best_global_tau = 0.5
    best_global_score = -1.0
    best_country_breakdown = {}
    
    for tau in threshold_grid:
        preds = {}
        for s1_id in ground_truth.keys():
            pairs = s1_pairs.get(s1_id, [])
            preds[s1_id] = {cid for cid, prob in pairs if prob >= tau}
            
        score, c_scores = compute_macro_f05(preds, ground_truth, s1_countries)
        if score > best_global_score:
            best_global_score = score
            best_global_tau = tau
            best_country_breakdown = c_scores
            
        print(f"Threshold tau={tau:.2f} -> Macro F_0.5 = {score*100:.3f}% | US: {c_scores.get('US',0)*100:.3f}% | India: {c_scores.get('India',0)*100:.3f}%")
        
    print(f"\n>> Best Global Threshold: tau={best_global_tau:.2f} with Macro F_0.5 = {best_global_score*100:.3f}%")
    
    # Per-country threshold search
    print("\n--- PER-COUNTRY INDEPENDENT TUNING ---")
    best_per_country_tau = {}
    for country in ["US", "India"]:
        c_s1_ids = {s1_id: true_set for s1_id, true_set in ground_truth.items() if s1_countries.get(s1_id) == country}
        best_c_tau = 0.5
        best_c_score = -1.0
        for tau in threshold_grid:
            c_preds = {}
            for s1_id in c_s1_ids.keys():
                pairs = s1_pairs.get(s1_id, [])
                c_preds[s1_id] = {cid for cid, prob in pairs if prob >= tau}
            score, _ = compute_macro_f05(c_preds, c_s1_ids)
            if score > best_c_score:
                best_c_score = score
                best_c_tau = tau
        best_per_country_tau[country] = (best_c_tau, best_c_score)
        print(f"[{country}] Best tau={best_c_tau:.2f} with Macro F_0.5 = {best_c_score*100:.3f}%")
        
    return best_global_tau, best_per_country_tau
