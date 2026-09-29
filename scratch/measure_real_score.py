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
    print("=== Testing Current Pipeline on Full Training Set Index ===")
    
    # Select 2,500 US and 2,500 India S1 holdout records
    val_s1 = {}
    with open('dataset/train/train_source1.tsv', 'r', encoding='utf-8') as f:
        next(f)
        us_cnt = 0
        in_cnt = 0
        for i, line in enumerate(f):
            if i < 200000: continue # Skip first 200k (which might overlap training)
            p = line.rstrip('\n').split('\t')
            ctry = p[3] if len(p) > 3 else 'Unknown'
            if ctry == 'US' and us_cnt < 2500:
                val_s1[p[0]] = prepare_record(p)
                us_cnt += 1
            elif ctry == 'India' and in_cnt < 2500:
                val_s1[p[0]] = prepare_record(p)
                in_cnt += 1
            if us_cnt >= 2500 and in_cnt >= 2500:
                break
                
    val_s1_ids = set(val_s1.keys())
    print(f"Loaded {len(val_s1)} validation S1 records ({us_cnt} US, {in_cnt} India)")
    
    # Load ground truth for these 5,000 records
    val_gt = {}
    with open('dataset/train/train_ground_truth.tsv', 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.rstrip('\n').split('\t')
            if p[0] in val_s1_ids:
                val_gt[p[0]] = [x.strip() for x in p[1].split(',') if x.strip()] if len(p) > 1 and p[1].strip() else []
                
    total_targets = sum(len(m) for m in val_gt.values())
    singletons = sum(1 for m in val_gt.values() if len(m) == 0)
    print(f"Validation targets: {total_targets}, Singletons: {singletons} ({singletons/len(val_s1)*100:.2f}%)")
    
    # We will test country by country
    clf = load_trained_model(MODEL_PATH)
    all_pred_matches = {}
    
    for ctry in ['US', 'India']:
        print(f"\n--- Testing Country: {ctry} ---")
        t0 = time.time()
        ctry_s1 = {k: v for k, v in val_s1.items() if v[2] == ctry}
        ctry_s1_ids = list(ctry_s1.keys())
        
        # Build FULL index for this country from train_source2 and train_source3
        inv_index = defaultdict(list)
        s23_records = {}
        for fn in ['dataset/train/train_source2.tsv', 'dataset/train/train_source3.tsv']:
            with open(fn, 'r', encoding='utf-8') as f:
                next(f)
                for line in f:
                    p = line.rstrip('\n').split('\t')
                    if len(p) > 3 and p[3] == ctry:
                        eid = p[0]
                        rec = prepare_record(p)
                        s23_records[eid] = rec
                        for k in get_blocking_keys(rec[0], rec[1], rec[2]):
                            inv_index[k].append(eid)
                            
        print(f"Indexed {len(s23_records)} records for {ctry} in {time.time()-t0:.2f}s. Unique keys: {len(inv_index)}")
        
        # Retrieve candidates and extract features
        t_cand = time.time()
        all_pair_feats = []
        all_pair_info = []
        cands_by_s1 = {}
        targets_found = 0
        ctry_targets = sum(len(val_gt[k]) for k in ctry_s1_ids)
        
        for s1_id in ctry_s1_ids:
            s1_rec = ctry_s1[s1_id]
            cands = retrieve_top_candidates(s1_rec[0], s1_rec[1], ctry, inv_index, top_k=TOP_CANDIDATES_PER_S1)
            cands_by_s1[s1_id] = cands
            gt_set = set(val_gt[s1_id])
            targets_found += len(gt_set & set(cands))
            
            for cid in cands:
                cand_rec = s23_records[cid]
                feat = extract_pair_features(s1_rec, cand_rec)
                all_pair_feats.append(feat)
                all_pair_info.append((s1_id, cid))
                
        rec_cand = targets_found / ctry_targets if ctry_targets > 0 else 0
        print(f"Candidate retrieval in {time.time()-t_cand:.2f}s. Blocking Recall: {targets_found}/{ctry_targets} ({rec_cand*100:.2f}%)")
        print(f"Total candidate pairs: {len(all_pair_feats)}")
        
        # Predict with XGBoost
        if len(all_pair_feats) > 0:
            X_eval = np.array(all_pair_feats, dtype=np.float32)
            probs = clf.predict_proba(X_eval)[:, 1]
            
            # 1-to-1 competitive matching at threshold
            for thresh in [0.70, 0.78, 0.85, 0.90]:
                cand_to_best_s1 = {}
                s1_cand_probs = defaultdict(dict)
                for (s1_id, cid), prob in zip(all_pair_info, probs):
                    if prob >= thresh:
                        s1_cand_probs[s1_id][cid] = prob
                        if cid not in cand_to_best_s1 or prob > cand_to_best_s1[cid][0]:
                            cand_to_best_s1[cid] = (prob, s1_id)
                            
                ctry_preds = {}
                for s1_id in ctry_s1_ids:
                    matched = []
                    for cid, prob in s1_cand_probs.get(s1_id, {}).items():
                        if cand_to_best_s1.get(cid, (None, None))[1] == s1_id:
                            matched.append(cid)
                    ctry_preds[s1_id] = matched
                    
                ctry_gt = {k: val_gt[k] for k in ctry_s1_ids}
                f05 = compute_macro_f05(ctry_gt, ctry_preds)
                tp = sum(len(set(ctry_gt[k]) & set(ctry_preds[k])) for k in ctry_s1_ids)
                tot_p = sum(len(ctry_preds[k]) for k in ctry_s1_ids)
                fp = tot_p - tp
                p_val = tp / tot_p if tot_p > 0 else 0
                r_val = tp / ctry_targets if ctry_targets > 0 else 0
                print(f"Threshold {thresh:.2f} -> Macro F0.5: {f05:.4f} | Prec: {p_val*100:.2f}% | Rec: {r_val*100:.2f}% | TP={tp}, FP={fp}")
                if thresh == MATCH_PROB_THRESHOLD:
                    all_pred_matches.update(ctry_preds)
                    
        del inv_index
        del s23_records
        
    overall_f05 = compute_macro_f05(val_gt, all_pred_matches)
    print(f"\n==========================================")
    print(f"OVERALL VALIDATION MACRO F0.5 (threshold={MATCH_PROB_THRESHOLD}): {overall_f05:.4f}")
    print(f"==========================================")

if __name__ == '__main__':
    main()
