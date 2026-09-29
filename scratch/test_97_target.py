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
from scratch.test_enhanced_model import extract_primary_house_number
from src.utils import extract_state, compute_macro_f05

DBA_REGEX = re.compile(r'\b(?:dba|d/b/a|ta|t/a|fka|formerly known as|operating as)\b', re.IGNORECASE)

def prepare_ultra_record(p: list) -> tuple:
    eid = p[0]
    name = p[1] if len(p) > 1 else ""
    addr = p[2] if len(p) > 2 else ""
    ctry = p[3] if len(p) > 3 else "Unknown"
    
    # Check for DBA / trade name
    dba_parts = [name]
    if DBA_REGEX.search(name):
        parts = DBA_REGEX.split(name)
        dba_parts.extend([p.strip() for p in parts if p.strip()])
        
    nc = clean_name(name)
    core = clean_core_name(name)
    concat_core = ''.join(core.split())
    ac = clean_addr(addr)
    all_nums = extract_all_numbers(addr)
    p_num = extract_primary_house_number(addr)
    pcode = extract_postal_code(addr, ctry)
    state = extract_state(addr, ctry)
    
    dba_cores = [clean_core_name(dp) for dp in dba_parts]
    
    return (name, addr, ctry, nc, core, concat_core, ac, all_nums, p_num, pcode, state, dba_cores)

def extract_ultra_features(s1_rec: tuple, c_rec: tuple) -> list:
    _, _, _, s1_nc, s1_core, s1_ccore, s1_ac, s1_nums, s1_pnum, s1_pcode, s1_state, s1_dbas = s1_rec
    _, _, _, c_nc, c_core, c_ccore, c_ac, c_nums, c_pnum, c_pcode, c_state, c_dbas = c_rec
    
    # 1. Core Name similarities
    core_sim = fuzz.token_sort_ratio(s1_core, c_core) / 100.0 if (s1_core and c_core) else 0.0
    core_exact = 1.0 if s1_core and c_core and s1_core == c_core else 0.0
    core_partial = fuzz.partial_ratio(s1_core, c_core) / 100.0 if (s1_core and c_core) else 0.0
    core_set = fuzz.token_set_ratio(s1_core, c_core) / 100.0 if (s1_core and c_core) else 0.0
    
    # 2. Concat core name similarity (matches urbanbakery vs urban bakery)
    ccore_sim = fuzz.ratio(s1_ccore, c_ccore) / 100.0 if (s1_ccore and c_ccore) else 0.0
    
    # 3. DBA / Sub-name cross matching
    max_dba_sim = core_sim
    for dp1 in s1_dbas:
        for dp2 in c_dbas:
            if dp1 and dp2:
                sim = fuzz.token_sort_ratio(dp1, dp2) / 100.0
                if sim > max_dba_sim: max_dba_sim = sim
                
    # 4. Brand features
    b_sim, b_exact, b_conflict, len_s1_b, len_c_b = compute_brand_features(s1_core, c_core)
    
    # 5. Primary house number
    house_match = 0.0
    house_conflict = 0.0
    if s1_pnum and c_pnum:
        if s1_pnum == c_pnum:
            house_match = 1.0
        elif s1_pnum in c_pnum or c_pnum in s1_pnum:
            house_match = 0.5
        else:
            house_conflict = 1.0
            
    # 6. All numbers compatibility
    has_num_c, num_conf, num_sub = check_number_compatibility(s1_nums, c_nums)
    
    # 7. Postal code consistency
    pcode_status = 0.0
    if s1_pcode and c_pcode:
        pcode_status = 1.0 if s1_pcode == c_pcode else -1.0
        
    # 8. Geographic state consistency
    state_status = 0.0
    if s1_state and c_state:
        state_status = 1.0 if s1_state == c_state else -1.0
        
    # 9. Address similarities
    has_s1_addr = 1.0 if s1_ac else 0.0
    has_c_addr = 1.0 if c_ac else 0.0
    addr_sim = fuzz.token_sort_ratio(s1_ac, c_ac) / 100.0 if (s1_ac and c_ac) else 0.0
    addr_partial = fuzz.partial_ratio(s1_ac, c_ac) / 100.0 if (s1_ac and c_ac) else 0.0
    
    # 10. Combined string similarity
    s1_comb = f"{s1_core} {s1_ac}"
    c_comb = f"{c_core} {c_ac}"
    comb_sim = fuzz.token_sort_ratio(s1_comb, c_comb) / 100.0
    
    # 11. High-precision rule flags
    addr_match_brand_conf = 1.0 if addr_sim >= 0.75 and b_conflict == 1.0 else 0.0
    brand_match_house_conf = 1.0 if (b_sim >= 0.8 or core_sim >= 0.8) and house_conflict == 1.0 else 0.0
    perfect_align = 1.0 if (b_exact == 1.0 or core_exact == 1.0 or ccore_sim >= 0.95) and (house_match == 1.0 or not (s1_pnum and c_pnum)) and state_status >= 0 else 0.0

    return [
        core_sim, core_exact, core_partial, core_set, ccore_sim, max_dba_sim,
        b_sim, b_exact, b_conflict,
        house_match, house_conflict,
        has_num_c, num_conf, num_sub,
        pcode_status, state_status,
        addr_sim, addr_partial, comb_sim,
        has_s1_addr, has_c_addr,
        addr_match_brand_conf, brand_match_house_conf, perfect_align
    ]

def main():
    print("=== Training Ultra ER Model with Hard Negatives & Evaluating Target > 0.97 ===")
    t0 = time.time()
    
    # Load 25,000 S1 records (20,000 train, 5,000 val)
    s1_dict = {}
    with open('dataset/train/train_source1.tsv', 'r', encoding='utf-8') as f:
        next(f)
        for i, line in enumerate(f):
            if i >= 25000: break
            p = line.rstrip('\n').split('\t')
            s1_dict[p[0]] = prepare_ultra_record(p)
            
    all_s1_ids = list(s1_dict.keys())
    train_s1_ids = all_s1_ids[:20000]
    val_s1_ids = all_s1_ids[20000:25000]
    
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
    print(f"Train targets: {len(train_targets)}, Val targets: {len(val_targets)}")
    
    # Index 800,000 S2/S3 records
    print("Indexing candidate pool...")
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
                elif cnt < 350000:
                    rec = prepare_ultra_record(p)
                    s23_records[eid] = rec
                    for k in get_enhanced_blocking_keys(p[1], p[2], p[3] if len(p) > 3 else 'Unknown'):
                        inv_index[k].append(eid)
                    cnt += 1
                    
    print(f"Indexed {len(s23_records)} candidate records in {time.time()-t0:.2f}s.")
    
    def get_top_cands(rec, top_k=40):
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

    # Mine training pairs
    print("Mining training pairs (positives + hard negatives)...")
    X_train = []
    y_train = []
    
    for s1_id in train_s1_ids:
        s1_rec = s1_dict[s1_id]
        gt_set = set(all_gt.get(s1_id, []))
        cands = get_top_cands(s1_rec, top_k=30)
        
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
    print(f"Training pairs: {len(X_train)} (Positives: {sum(y_train)}, Negatives: {len(y_train)-sum(y_train)})")
    
    print("Training XGBoost...")
    t_tr = time.time()
    clf = xgb.XGBClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric='logloss',
        random_state=42
    )
    clf.fit(X_train, y_train)
    print(f"XGBoost trained in {time.time()-t_tr:.2f}s")
    
    # Evaluate on 5,000 Holdout S1 records
    print("\n--- Evaluating on 5,000 Holdout S1 Records ---")
    val_pair_feats = []
    val_pair_info = []
    targets_retrieved = 0
    total_val_targets = len(val_targets)
    
    for s1_id in val_s1_ids:
        s1_rec = s1_dict[s1_id]
        cands = get_top_cands(s1_rec, top_k=40)
        gt_set = set(all_gt.get(s1_id, []))
        targets_retrieved += len(gt_set & set(cands))
        
        for cid in cands:
            if cid in s23_records:
                val_pair_feats.append(extract_ultra_features(s1_rec, s23_records[cid]))
                val_pair_info.append((s1_id, cid))
                
    print(f"Blocking Recall @ top-40: {targets_retrieved}/{total_val_targets} ({targets_retrieved/total_val_targets*100:.2f}%)")
    
    X_val = np.array(val_pair_feats, dtype=np.float32)
    probs = clf.predict_proba(X_val)[:, 1]
    val_gt_dict = {k: all_gt[k] for k in val_s1_ids}
    
    for thresh in [0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.94]:
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
