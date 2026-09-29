import sys
import os
import re
import math
import time
import unicodedata
from collections import defaultdict, Counter
import numpy as np
from rapidfuzz import fuzz, distance
import xgboost as xgb

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')
sys.path.insert(0, 'code/business_entity_resolution')

from scratch.test_enhanced_features import (
    clean_name, clean_core_name, clean_addr, extract_all_numbers,
    extract_postal_code, LEGAL_SUFFIXES, COMMON_ADDR_WORDS
)
from scratch.test_enhanced_blocking import get_enhanced_blocking_keys
from scratch.test_97_target import prepare_ultra_record, extract_ultra_features
from src.utils import compute_macro_f05

def main():
    print("=== Final Pipeline Benchmark: IDF Blocking + Ultra Features + Hard Negatives ===")
    t0 = time.time()
    
    # 1. Load 30,000 S1 records (25,000 train, 5,000 holdout validation)
    s1_dict = {}
    with open('dataset/train/train_source1.tsv', 'r', encoding='utf-8') as f:
        next(f)
        for i, line in enumerate(f):
            if i >= 30000: break
            p = line.rstrip('\n').split('\t')
            s1_dict[p[0]] = prepare_ultra_record(p)
            
    all_s1_ids = list(s1_dict.keys())
    train_s1_ids = all_s1_ids[:25000]
    val_s1_ids = all_s1_ids[25000:30000]
    
    all_gt = {}
    with open('dataset/train/train_ground_truth.tsv', 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.rstrip('\n').split('\t')
            if p[0] in s1_dict:
                all_gt[p[0]] = [x.strip() for x in p[1].split(',') if x.strip()] if len(p) > 1 and p[1].strip() else []
                
    train_targets = {m for eid in train_s1_ids for m in all_gt.get(eid, [])}
    val_targets = {m for eid in val_s1_ids for m in all_gt.get(eid, [])}
    needed_targets = train_targets | val_targets
    print(f"Loaded records. Train targets: {len(train_targets)}, Val targets: {len(val_targets)}")
    
    # 2. Index 1,000,000 S2/S3 candidates
    print("Building IDF-weighted inverted index with 1,000,000 candidate records...")
    inv_index = defaultdict(list)
    s23_records = {}
    
    for fn in ['dataset/train/train_source2.tsv', 'dataset/train/train_source3.tsv']:
        with open(fn, 'r', encoding='utf-8') as f:
            next(f)
            cnt = 0
            for line in f:
                p = line.rstrip('\n').split('\t')
                eid = p[0]
                if eid in needed_targets:
                    rec = prepare_ultra_record(p)
                    s23_records[eid] = rec
                    for k in get_enhanced_blocking_keys(p[1], p[2], p[3] if len(p) > 3 else 'Unknown'):
                        inv_index[k].append(eid)
                elif cnt < 450000:
                    rec = prepare_ultra_record(p)
                    s23_records[eid] = rec
                    for k in get_enhanced_blocking_keys(p[1], p[2], p[3] if len(p) > 3 else 'Unknown'):
                        inv_index[k].append(eid)
                    cnt += 1
                    
    print(f"Indexed {len(s23_records)} candidates in {time.time()-t0:.2f}s. Unique keys: {len(inv_index)}")
    
    def get_top_cands_idf(rec, top_k=30):
        name, addr, ctry = rec[0], rec[1], rec[2]
        cands = Counter()
        for k in get_enhanced_blocking_keys(name, addr, ctry):
            postings = inv_index.get(k)
            if not postings: continue
            n_p = len(postings)
            if n_p <= 2500:
                k_type = k.split(':')[1] if ':' in k else ''
                base_w = 20.0 if k_type in ('nn', 'ncc') else (14.0 if k_type == 'np' else (10.0 if k_type == 'ns' else 4.0))
                idf = base_w / math.log(2 + n_p)
                for cid in postings:
                    cands[cid] += idf
        return [c for c, _ in cands.most_common(top_k)]

    # 3. Mine training pairs (all positives + 10 hard negatives per S1)
    print("Mining training pairs (positives + hard negatives from IDF blocking)...")
    X_train = []
    y_train = []
    
    for s1_id in train_s1_ids:
        s1_rec = s1_dict[s1_id]
        gt_set = set(all_gt.get(s1_id, []))
        cands = get_top_cands_idf(s1_rec, top_k=25)
        
        for tid in gt_set:
            if tid in s23_records:
                X_train.append(extract_ultra_features(s1_rec, s23_records[tid]))
                y_train.append(1)
                
        neg_count = 0
        for cid in cands:
            if cid not in gt_set and cid in s23_records:
                X_train.append(extract_ultra_features(s1_rec, s23_records[cid]))
                y_train.append(0)
                neg_count += 1
                if neg_count >= 10: break
                
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    print(f"Constructed {len(X_train)} training pairs (Positives: {sum(y_train)}, Negatives: {len(y_train)-sum(y_train)})")
    
    # 4. Train XGBoost Model
    print("Training XGBoost...")
    t_tr = time.time()
    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric='logloss',
        random_state=42
    )
    clf.fit(X_train, y_train)
    print(f"XGBoost trained in {time.time()-t_tr:.2f}s")
    
    # 5. Evaluate on 5,000 Holdout Validation Records
    print("\n--- Evaluating on 5,000 Holdout S1 Records ---")
    val_pair_feats = []
    val_pair_info = []
    targets_retrieved = 0
    total_val_targets = len(val_targets)
    
    for s1_id in val_s1_ids:
        s1_rec = s1_dict[s1_id]
        cands = get_top_cands_idf(s1_rec, top_k=30)
        gt_set = set(all_gt.get(s1_id, []))
        targets_retrieved += len(gt_set & set(cands))
        
        for cid in cands:
            if cid in s23_records:
                val_pair_feats.append(extract_ultra_features(s1_rec, s23_records[cid]))
                val_pair_info.append((s1_id, cid))
                
    rec_b = targets_retrieved / total_val_targets if total_val_targets > 0 else 0
    print(f"IDF Blocking Recall on Validation: {targets_retrieved}/{total_val_targets} ({rec_b*100:.2f}%)")
    print(f"Scoring {len(val_pair_feats)} candidate pairs...")
    
    X_val = np.array(val_pair_feats, dtype=np.float32)
    probs = clf.predict_proba(X_val)[:, 1]
    val_gt_dict = {k: all_gt[k] for k in val_s1_ids}
    
    for thresh in [0.70, 0.75, 0.78, 0.80, 0.82, 0.85, 0.88, 0.90]:
        cand_to_best_s1 = {}
        s1_cand_probs = defaultdict(dict)
        for (s1_id, cid), prob in zip(val_pair_info, probs):
            if prob >= thresh:
                s1_cand_probs[s1_id][cid] = prob
                if cid not in cand_to_best_s1 or prob > cand_to_best_s1[cid][0]:
                    cand_to_best_s1[cid] = (prob, s1_id)
                    
        pred_matches = {}
        for s1_id in val_s1_ids:
            matched = []
            for cid, prob in s1_cand_probs.get(s1_id, {}).items():
                if cand_to_best_s1.get(cid, (None, None))[1] == s1_id:
                    matched.append(cid)
            pred_matches[s1_id] = matched
            
        f05 = compute_macro_f05(val_gt_dict, pred_matches)
        tp = sum(len(set(val_gt_dict[k]) & set(pred_matches[k])) for k in val_s1_ids)
        total_p = sum(len(pred_matches[k]) for k in val_s1_ids)
        fp = total_p - tp
        fn = total_val_targets - tp
        p_val = tp / total_p if total_p > 0 else 0
        r_val = tp / total_val_targets if total_val_targets > 0 else 0
        print(f"Threshold {thresh:.2f} -> Macro F0.5: {f05:.4f} | Prec: {p_val*100:.2f}% | Rec: {r_val*100:.2f}% | TP={tp}, FP={fp}, FN={fn}")

if __name__ == '__main__':
    main()
