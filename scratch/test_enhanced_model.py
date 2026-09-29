import sys
import os
import re
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
from scratch.test_enhanced_blocking import (
    get_enhanced_blocking_keys, KEY_WEIGHTS, MAX_POSTINGS
)
from scratch.test_brand_logic import (
    extract_brand_words, compute_brand_features, check_number_compatibility
)
from src.utils import extract_state, compute_macro_f05

def extract_primary_house_number(addr: str) -> str:
    if not addr: return None
    m = re.search(r'^\s*#*\s*(\d+)', addr)
    if m: return m.group(1).lstrip('0') or '0'
    m = re.search(r'\b(?:no|door|plot|flat|h\.no|wz|b|d|ews|pl)\s*[:#.-]?\s*(\d+)', addr, re.IGNORECASE)
    if m: return m.group(1).lstrip('0') or '0'
    m = re.search(r'\b\d+\b', addr)
    if m: return m.group(0).lstrip('0') or '0'
    return None

def prepare_enhanced_record(p: list) -> tuple:
    eid = p[0]
    name = p[1] if len(p) > 1 else ""
    addr = p[2] if len(p) > 2 else ""
    ctry = p[3] if len(p) > 3 else "Unknown"
    
    nc = clean_name(name)
    core = clean_core_name(name)
    ac = clean_addr(addr)
    all_nums = extract_all_numbers(addr)
    p_num = extract_primary_house_number(addr)
    pcode = extract_postal_code(addr, ctry)
    state = extract_state(addr, ctry)
    
    return (name, addr, ctry, nc, core, ac, all_nums, p_num, pcode, state)

def extract_comprehensive_features(s1_rec: tuple, c_rec: tuple) -> list:
    _, _, _, s1_nc, s1_core, s1_ac, s1_nums, s1_pnum, s1_pcode, s1_state = s1_rec
    _, _, _, c_nc, c_core, c_ac, c_nums, c_pnum, c_pcode, c_state = c_rec
    
    # 1. Core Name similarities
    core_sim = fuzz.token_sort_ratio(s1_core, c_core) / 100.0 if (s1_core and c_core) else 0.0
    core_exact = 1.0 if s1_core and c_core and s1_core == c_core else 0.0
    core_partial = fuzz.partial_ratio(s1_core, c_core) / 100.0 if (s1_core and c_core) else 0.0
    core_set = fuzz.token_set_ratio(s1_core, c_core) / 100.0 if (s1_core and c_core) else 0.0
    
    # 2. Brand features
    b_sim, b_exact, b_conflict, len_s1_b, len_c_b = compute_brand_features(s1_core, c_core)
    
    # 3. Primary house number
    house_match = 0.0
    house_conflict = 0.0
    if s1_pnum and c_pnum:
        if s1_pnum == c_pnum:
            house_match = 1.0
        elif s1_pnum in c_pnum or c_pnum in s1_pnum:
            house_match = 0.5
        else:
            house_conflict = 1.0
            
    # 4. All numbers compatibility
    has_num_c, num_conf, num_sub = check_number_compatibility(s1_nums, c_nums)
    
    # 5. Postal code consistency
    pcode_status = 0.0
    if s1_pcode and c_pcode:
        pcode_status = 1.0 if s1_pcode == c_pcode else -1.0
        
    # 6. Geographic state consistency
    state_status = 0.0
    if s1_state and c_state:
        state_status = 1.0 if s1_state == c_state else -1.0
        
    # 7. Address similarities
    has_s1_addr = 1.0 if s1_ac else 0.0
    has_c_addr = 1.0 if c_ac else 0.0
    addr_sim = fuzz.token_sort_ratio(s1_ac, c_ac) / 100.0 if (s1_ac and c_ac) else 0.0
    addr_partial = fuzz.partial_ratio(s1_ac, c_ac) / 100.0 if (s1_ac and c_ac) else 0.0
    
    # 8. Combined string similarity
    s1_comb = f"{s1_core} {s1_ac}"
    c_comb = f"{c_core} {c_ac}"
    comb_sim = fuzz.token_sort_ratio(s1_comb, c_comb) / 100.0
    
    # 9. Synergistic interaction features
    # Address match but brand conflict (same building, different business)
    addr_match_brand_conf = 1.0 if addr_sim >= 0.75 and b_conflict == 1.0 else 0.0
    # Brand match but house number conflict (same street, different building)
    brand_match_house_conf = 1.0 if b_sim >= 0.8 and house_conflict == 1.0 else 0.0
    # Perfect alignment
    perfect_align = 1.0 if (b_exact == 1.0 or core_exact == 1.0) and (house_match == 1.0 or not (s1_pnum and c_pnum)) and state_status >= 0 else 0.0

    return [
        core_sim, core_exact, core_partial, core_set,
        b_sim, b_exact, b_conflict,
        house_match, house_conflict,
        has_num_c, num_conf, num_sub,
        pcode_status, state_status,
        addr_sim, addr_partial, comb_sim,
        has_s1_addr, has_c_addr,
        addr_match_brand_conf, brand_match_house_conf, perfect_align
    ]

def main():
    print("=== Enhanced Model: Training with Real Hard Negatives & Evaluation ===")
    t0 = time.time()
    
    # Load 15,000 S1 records (10,000 train, 2,500 val)
    s1_dict = {}
    with open('dataset/train/train_source1.tsv', 'r', encoding='utf-8') as f:
        next(f)
        for i, line in enumerate(f):
            if i >= 15000: break
            p = line.rstrip('\n').split('\t')
            s1_dict[p[0]] = prepare_enhanced_record(p)
            
    all_s1_ids = list(s1_dict.keys())
    train_s1_ids = all_s1_ids[:10000]
    val_s1_ids = all_s1_ids[10000:12500]
    
    # Load ground truth
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
    print(f"Loaded S1 records. Train targets: {len(train_targets)}, Val targets: {len(val_targets)}")
    
    # Index 600,000 S2/S3 records (all targets + 500,000 background records)
    print("Building blocking index with 600,000 candidates...")
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
                    rec = prepare_enhanced_record(p)
                    s23_records[eid] = rec
                    for k in get_enhanced_blocking_keys(p[1], p[2], p[3] if len(p) > 3 else 'Unknown'):
                        inv_index[k].append(eid)
                elif cnt < 250000:
                    rec = prepare_enhanced_record(p)
                    s23_records[eid] = rec
                    for k in get_enhanced_blocking_keys(p[1], p[2], p[3] if len(p) > 3 else 'Unknown'):
                        inv_index[k].append(eid)
                    cnt += 1
                    
    print(f"Indexed {len(s23_records)} records in {time.time()-t0:.2f}s. Unique keys: {len(inv_index)}")
    
    # Helper to retrieve top candidates
    def get_top_cands(rec, top_k=30):
        name, addr, ctry = rec[0], rec[1], rec[2]
        cands = Counter()
        for k in get_enhanced_blocking_keys(name, addr, ctry):
            k_type = k.split(':')[1] if ':' in k else ''
            max_p = MAX_POSTINGS.get(k_type, 150)
            postings = inv_index.get(k)
            if postings and len(postings) <= max_p:
                w = KEY_WEIGHTS.get(k_type, 1)
                for cid in postings:
                    cands[cid] += w
        return [c for c, _ in cands.most_common(top_k)]

    # 1. Mine Training Pairs (Positive + Realistic Hard Negatives from blocking)
    print("Mining training pairs (positives + hard negatives)...")
    X_train = []
    y_train = []
    
    for s1_id in train_s1_ids:
        s1_rec = s1_dict[s1_id]
        gt_set = set(all_gt.get(s1_id, []))
        cands = get_top_cands(s1_rec, top_k=25)
        
        # Include all ground truth targets that exist in index as positives
        for tid in gt_set:
            if tid in s23_records:
                feat = extract_comprehensive_features(s1_rec, s23_records[tid])
                X_train.append(feat)
                y_train.append(1)
                
        # Include up to 8 hard negatives retrieved by blocking
        neg_count = 0
        for cid in cands:
            if cid not in gt_set and cid in s23_records:
                feat = extract_comprehensive_features(s1_rec, s23_records[cid])
                X_train.append(feat)
                y_train.append(0)
                neg_count += 1
                if neg_count >= 8: break
                
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    print(f"Constructed {len(X_train)} training pairs (Positives: {sum(y_train)}, Hard Negatives: {len(y_train)-sum(y_train)})")
    
    # 2. Train XGBoost Model
    print("Training XGBoost with hard negatives...")
    t_tr = time.time()
    clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1.0,
        eval_metric='logloss',
        random_state=42
    )
    clf.fit(X_train, y_train)
    print(f"Model trained in {time.time()-t_tr:.2f}s")
    
    # 3. Evaluate on Validation Set
    print("\n--- Evaluating on 2,500 Holdout S1 Records ---")
    val_cands_map = {}
    val_pair_feats = []
    val_pair_info = []
    targets_retrieved = 0
    total_val_targets = len(val_targets)
    
    for s1_id in val_s1_ids:
        s1_rec = s1_dict[s1_id]
        cands = get_top_cands(s1_rec, top_k=30)
        val_cands_map[s1_id] = cands
        gt_set = set(all_gt.get(s1_id, []))
        targets_retrieved += len(gt_set & set(cands))
        
        for cid in cands:
            if cid in s23_records:
                feat = extract_comprehensive_features(s1_rec, s23_records[cid])
                val_pair_feats.append(feat)
                val_pair_info.append((s1_id, cid))
                
    print(f"Blocking Recall on Validation: {targets_retrieved}/{total_val_targets} ({targets_retrieved/total_val_targets*100:.2f}%)")
    print(f"Scoring {len(val_pair_feats)} candidate pairs...")
    
    X_val = np.array(val_pair_feats, dtype=np.float32)
    probs = clf.predict_proba(X_val)[:, 1]
    
    val_gt_dict = {k: all_gt[k] for k in val_s1_ids}
    
    for thresh in [0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.93]:
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
