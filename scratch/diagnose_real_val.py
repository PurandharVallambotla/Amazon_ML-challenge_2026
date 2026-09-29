import sys
import os
import time
from collections import defaultdict, Counter
import numpy as np

sys.path.insert(0, 'code/business_entity_resolution')
from src.pipeline import prepare_record
from src.blocking import get_blocking_keys, retrieve_top_candidates
from src.features import extract_pair_features
from src.model import load_trained_model
from src.utils import compute_macro_f05
from src.config import MODEL_PATH, MATCH_PROB_THRESHOLD, TOP_CANDIDATES_PER_S1

def main():
    print("=== Diagnostic: Evaluating Pipeline Performance on Validation Set ===")
    
    # 1. Load 3,000 S1 validation records (US & India)
    val_s1 = {}
    with open('dataset/train/train_source1.tsv', 'r', encoding='utf-8') as fp:
        next(fp)
        for i, line in enumerate(fp):
            if 100000 <= i < 103000:
                p = line.rstrip('\n').split('\t')
                val_s1[p[0]] = prepare_record(p)
            elif i >= 103000:
                break
                
    val_s1_ids = set(val_s1.keys())
    val_gt = {}
    with open('dataset/train/train_ground_truth.tsv', 'r', encoding='utf-8') as fp:
        next(fp)
        for line in fp:
            p = line.rstrip('\n').split('\t')
            if p[0] in val_s1_ids:
                val_gt[p[0]] = [x.strip() for x in p[1].split(',') if x.strip()] if len(p) > 1 and p[1].strip() else []
                
    val_targets = {m for mlist in val_gt.values() for m in mlist}
    print(f"Loaded {len(val_s1)} validation S1 records with {len(val_targets)} true targets.")
    
    # 2. Build index of S2 and S3: all true targets + 300,000 background records
    print("Indexing S2 and S3 (including 300,000 background records)...")
    t0 = time.time()
    s23_records = {}
    inv_index = defaultdict(list)
    
    for src_file in ['dataset/train/train_source2.tsv', 'dataset/train/train_source3.tsv']:
        with open(src_file, 'r', encoding='utf-8') as fp:
            next(fp)
            cnt = 0
            for line in fp:
                p = line.rstrip('\n').split('\t')
                eid = p[0]
                if eid in val_targets:
                    rec = prepare_record(p)
                    s23_records[eid] = rec
                    for k in get_blocking_keys(rec[0], rec[1], rec[2]):
                        inv_index[k].append(eid)
                elif cnt < 150000:
                    rec = prepare_record(p)
                    s23_records[eid] = rec
                    for k in get_blocking_keys(rec[0], rec[1], rec[2]):
                        inv_index[k].append(eid)
                    cnt += 1
                    
    print(f"Indexed {len(s23_records)} records in {time.time()-t0:.2f}s. Unique keys: {len(inv_index)}")
    
    # 3. Check Blocking Recall
    print("\n--- 1. Evaluating Blocking Recall ---")
    retrieved_cands = {}
    total_targets = len(val_targets)
    targets_retrieved = 0
    
    for s1_id, s1_rec in val_s1.items():
        name, addr, ctry = s1_rec[0], s1_rec[1], s1_rec[2]
        cands = retrieve_top_candidates(name, addr, ctry, inv_index, top_k=TOP_CANDIDATES_PER_S1)
        retrieved_cands[s1_id] = cands
        gt_set = set(val_gt.get(s1_id, []))
        targets_retrieved += len(gt_set & set(cands))
        
    blocking_recall = targets_retrieved / total_targets if total_targets > 0 else 0
    print(f"Blocking Recall @ top-{TOP_CANDIDATES_PER_S1}: {targets_retrieved}/{total_targets} ({blocking_recall*100:.2f}%)")
    
    # Check with top-30 and top-50
    for k in [25, 40]:
        t_ret = 0
        for s1_id, s1_rec in val_s1.items():
            name, addr, ctry = s1_rec[0], s1_rec[1], s1_rec[2]
            cands = retrieve_top_candidates(name, addr, ctry, inv_index, top_k=k)
            gt_set = set(val_gt.get(s1_id, []))
            t_ret += len(gt_set & set(cands))
        print(f"Blocking Recall @ top-{k}: {t_ret}/{total_targets} ({t_ret/total_targets*100:.2f}%)")
        
    # 4. Check XGBoost Scoring and Thresholds
    print("\n--- 2. Evaluating Model & Matching Decisions ---")
    clf = load_trained_model(MODEL_PATH)
    
    all_pair_feats = []
    all_pair_info = []
    for s1_id, s1_rec in val_s1.items():
        for cid in retrieved_cands[s1_id]:
            if cid in s23_records:
                cand_rec = s23_records[cid]
                feat = extract_pair_features(s1_rec, cand_rec)
                all_pair_feats.append(feat)
                all_pair_info.append((s1_id, cid))
                
    X_val = np.array(all_pair_feats, dtype=np.float32)
    probs = clf.predict_proba(X_val)[:, 1]
    
    # Check distribution of probs on true pairs vs false pairs
    true_probs = []
    false_probs = []
    for (s1_id, cid), prob in zip(all_pair_info, probs):
        if cid in val_gt.get(s1_id, []):
            true_probs.append(prob)
        else:
            false_probs.append(prob)
            
    print(f"True pairs scored: {len(true_probs)}, Mean prob: {np.mean(true_probs):.4f}, Median: {np.median(true_probs):.4f}")
    print(f"False pairs scored: {len(false_probs)}, Mean prob: {np.mean(false_probs):.4f}, Median: {np.median(false_probs):.4f}")
    print(f"True pairs with prob < 0.78: {sum(1 for p in true_probs if p < 0.78)} ({sum(1 for p in true_probs if p < 0.78)/len(true_probs)*100:.2f}%)")
    print(f"False pairs with prob >= 0.78: {sum(1 for p in false_probs if p >= 0.78)}")
    
    # 5. Evaluate Macro F0.5 across different thresholds
    print("\n--- 3. Macro F0.5 across Probability Thresholds ---")
    for thresh in [0.40, 0.50, 0.60, 0.70, 0.75, 0.78, 0.80, 0.85, 0.90]:
        cand_to_best_s1 = {}
        s1_cand_probs = defaultdict(dict)
        for (s1_id, cid), prob in zip(all_pair_info, probs):
            if prob >= thresh:
                s1_cand_probs[s1_id][cid] = prob
                if cid not in cand_to_best_s1 or prob > cand_to_best_s1[cid][0]:
                    cand_to_best_s1[cid] = (prob, s1_id)
                    
        pred_matches = {}
        for s1_id in val_s1:
            matched = []
            for cid, prob in s1_cand_probs.get(s1_id, {}).items():
                if cand_to_best_s1.get(cid, (None, None))[1] == s1_id:
                    matched.append(cid)
            pred_matches[s1_id] = matched
            
        f05 = compute_macro_f05(val_gt, pred_matches)
        
        # Calculate overall TP, FP, FN
        tp = sum(len(set(val_gt[k]) & set(pred_matches[k])) for k in val_s1)
        total_p = sum(len(pred_matches[k]) for k in val_s1)
        fp = total_p - tp
        fn = total_targets - tp
        p_val = tp / total_p if total_p > 0 else 0
        r_val = tp / total_targets if total_targets > 0 else 0
        print(f"Threshold {thresh:.2f} -> Macro F0.5: {f05:.4f} | Prec: {p_val*100:.2f}% | Rec: {r_val*100:.2f}% (TP={tp}, FP={fp}, FN={fn})")

if __name__ == '__main__':
    main()
