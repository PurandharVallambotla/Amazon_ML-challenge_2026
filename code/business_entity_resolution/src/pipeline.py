"""End-to-end Entity Resolution pipeline: training and inference.

Strictly country-agnostic and self-contained with zero external dependencies.
"""

import os
import re
import sys
import time
import math
from collections import defaultdict, Counter
from rapidfuzz import fuzz
import numpy as np

from src.config import (
    MODEL_PATH, DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR,
    MATCH_PROB_THRESHOLD, TOP_CANDIDATES_PER_S1, MAX_POSTINGS_PER_KEY
)
from src.utils import (
    clean_name, clean_core_name, clean_addr,
    extract_all_numbers, extract_primary_house_number,
    extract_geo_tokens, compute_macro_f05
)
from src.blocking import get_blocking_keys, retrieve_top_candidates
from src.features import extract_pair_features
from src.model import train_and_save_model, load_trained_model, predict_match_probabilities

DBA_REGEX = re.compile(r'\b(?:dba|d/b/a|ta|t/a|fka|formerly known as|operating as)\b', re.IGNORECASE)

def prepare_record(p: list) -> tuple:
    """Preprocess raw row [entity_id, business_name, business_address, country]."""
    eid = p[0]
    name = p[1] if len(p) > 1 else ""
    addr = p[2] if len(p) > 2 else ""
    ctry = p[3].strip().upper() if len(p) > 3 and p[3].strip() else "GLOBAL"
    
    # Extract trade / DBA name variations
    dba_parts = [name]
    if DBA_REGEX.search(name):
        parts = DBA_REGEX.split(name)
        dba_parts.extend([pt.strip() for pt in parts if pt.strip()])
        
    nc = clean_name(name)
    core = clean_core_name(name)
    concat_core = ''.join(core.split())
    ac = clean_addr(addr)
    all_nums = extract_all_numbers(addr)
    p_num = extract_primary_house_number(addr)
    pcodes = {n for n in all_nums if 4 <= len(n) <= 8}
    geo = extract_geo_tokens(addr)
    dba_cores = [clean_core_name(dp) for dp in dba_parts]
    
    return (name, addr, ctry, nc, core, concat_core, ac, all_nums, p_num, pcodes, geo, dba_cores)

def train_pipeline(data_dir: str = DEFAULT_DATA_DIR, sample_s1: int = 35000):
    """
    Train XGBoost model on representative training sample.
    Extracts positive pairs from ground truth and hard negative candidates from blocking.
    """
    print(f"--- Training Pipeline (Sample size: {sample_s1} S1 records) ---")
    s1_train_path = os.path.join(data_dir, "train", "train_source1.tsv")
    gt_path = os.path.join(data_dir, "train", "train_ground_truth.tsv")
    s2_path = os.path.join(data_dir, "train", "train_source2.tsv")
    s3_path = os.path.join(data_dir, "train", "train_source3.tsv")
    
    # 1. Load S1 sample
    t0 = time.time()
    s1_sample = {}
    with open(s1_train_path, 'r', encoding='utf-8') as fp:
        next(fp)
        for i, line in enumerate(fp):
            if i >= sample_s1:
                break
            p = line.rstrip('\n').split('\t')
            s1_sample[p[0]] = prepare_record(p)
    print(f"Loaded {len(s1_sample)} S1 records in {time.time()-t0:.2f}s")
    
    # 2. Load ground truth
    s1_ids = set(s1_sample.keys())
    gt_map = {}
    with open(gt_path, 'r', encoding='utf-8') as fp:
        next(fp)
        for line in fp:
            p = line.rstrip('\n').split('\t')
            if p[0] in s1_ids:
                gt_map[p[0]] = [x.strip() for x in p[1].split(',') if x.strip()] if len(p) > 1 and p[1].strip() else []
                
    all_target_ids = {m for mlist in gt_map.values() for m in mlist}
    print(f"Total true match targets in sample: {len(all_target_ids)}")
    
    # 3. Index S2 and S3 (all targets + 1,000,000 background records to mine hard negatives)
    s23_records = {}
    inv_index = defaultdict(list)
    
    for src_path, max_distractors in [(s2_path, 500000), (s3_path, 500000)]:
        dist_count = 0
        with open(src_path, 'r', encoding='utf-8') as fp:
            next(fp)
            for line in fp:
                p = line.rstrip('\n').split('\t')
                eid = p[0]
                if eid in all_target_ids:
                    rec = prepare_record(p)
                    s23_records[eid] = rec
                    for k in get_blocking_keys(rec[0], rec[1], rec[2]):
                        inv_index[k].append(eid)
                elif dist_count < max_distractors:
                    rec = prepare_record(p)
                    s23_records[eid] = rec
                    for k in get_blocking_keys(rec[0], rec[1], rec[2]):
                        inv_index[k].append(eid)
                    dist_count += 1
                    
    print(f"Indexed {len(s23_records)} candidate records. Unique blocking keys: {len(inv_index)}")
    
    # 4. Generate pairs and feature vectors
    print("Mining training pairs (positives + realistic hard negatives)...")
    X_train = []
    y_train = []
    
    for s1_id, s1_rec in s1_sample.items():
        name, addr, country = s1_rec[0], s1_rec[1], s1_rec[2]
        cands = retrieve_top_candidates(name, addr, country, inv_index, top_k=25)
        gt_set = set(gt_map.get(s1_id, []))
        
        # Positives
        for tid in gt_set:
            if tid in s23_records:
                feat = extract_pair_features(s1_rec, s23_records[tid])
                X_train.append(feat)
                y_train.append(1)
                
        # Hard Negatives
        neg_count = 0
        for cid in cands:
            if cid not in gt_set and cid in s23_records:
                feat = extract_pair_features(s1_rec, s23_records[cid])
                X_train.append(feat)
                y_train.append(0)
                neg_count += 1
                if neg_count >= 10:
                    break
                
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    print(f"Constructed {len(X_train)} training pairs (Positives: {sum(y_train)}, Hard Negatives: {len(y_train)-sum(y_train)})")
    
    # 5. Train and persist model
    clf = train_and_save_model(X_train, y_train, save_path=MODEL_PATH)
    return clf

def run_inference(test_dir: str = os.path.join(DEFAULT_DATA_DIR, "test"),
                  output_dir: str = DEFAULT_OUTPUT_DIR,
                  model_path: str = MODEL_PATH,
                  threshold: float = MATCH_PROB_THRESHOLD):
    """
    Run end-to-end candidate generation and entity matching on the test set.
    Outputs:
      - output/candidate_pairs.tsv
      - output/matching_results.tsv
    """
    print("=== Starting Test Set Inference ===")
    t_start = time.time()
    
    s1_file = os.path.join(test_dir, "test_source1.tsv")
    s2_file = os.path.join(test_dir, "test_source2.tsv")
    s3_file = os.path.join(test_dir, "test_source3.tsv")
    
    os.makedirs(output_dir, exist_ok=True)
    matching_out_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_out_path = os.path.join(output_dir, "candidate_pairs.tsv")
    
    # 1. Load trained model
    clf = load_trained_model(model_path)
    print("Loaded trained XGBoost model.")
    
    # 2. Inspect Source 1 entities and group dynamically by country label
    print("Scanning test_source1.tsv...")
    s1_all_ids = []
    s1_by_country = defaultdict(list)
    s1_records_by_country = defaultdict(dict)
    
    with open(s1_file, 'r', encoding='utf-8') as fp:
        next(fp)
        for line in fp:
            line_str = line.rstrip('\n')
            if not line_str:
                continue
            p = line_str.split('\t')
            eid = p[0]
            country = p[3].strip().upper() if len(p) > 3 and p[3].strip() else "GLOBAL"
            s1_all_ids.append(eid)
            s1_by_country[country].append(eid)
            s1_records_by_country[country][eid] = prepare_record(p)
            
    total_s1 = len(s1_all_ids)
    print(f"Total Source 1 test records: {total_s1}")
    for ctry, ids in s1_by_country.items():
        print(f"  Country '{ctry}': {len(ids)} records")
        
    final_candidates = {}
    final_matches = {}
    
    # 3. Process country by country to keep memory strictly bounded
    for ctry in list(s1_by_country.keys()):
        print(f"\n--- Processing Country: {ctry} ---")
        t_c = time.time()
        
        # Load S2 and S3 for this country
        inv_index = defaultdict(list)
        s23_records = {}
        
        for src_path in [s2_file, s3_file]:
            with open(src_path, 'r', encoding='utf-8') as fp:
                next(fp)
                for line in fp:
                    line_str = line.rstrip('\n')
                    if not line_str:
                        continue
                    p = line_str.split('\t')
                    rec_ctry = p[3].strip().upper() if len(p) > 3 and p[3].strip() else "GLOBAL"
                    if rec_ctry == ctry:
                        eid = p[0]
                        rec = prepare_record(p)
                        s23_records[eid] = rec
                        for k in get_blocking_keys(rec[0], rec[1], rec[2]):
                            inv_index[k].append(eid)
                            
        print(f"Loaded {len(s23_records)} candidate records for {ctry} in {time.time()-t_c:.2f}s. Unique keys: {len(inv_index)}")
        
        # Candidate retrieval and feature extraction
        t_cand = time.time()
        s1_country_records = s1_records_by_country[ctry]
        
        all_pair_feats = []
        all_pair_info = [] # (s1_id, cid)
        
        for s1_id in s1_by_country[ctry]:
            s1_rec = s1_country_records[s1_id]
            name, addr = s1_rec[0], s1_rec[1]
            cands = retrieve_top_candidates(name, addr, ctry, inv_index, top_k=TOP_CANDIDATES_PER_S1)
            final_candidates[s1_id] = cands
            
            s1_core = s1_rec[4]
            s1_ac = s1_rec[6]
            
            for cid in cands:
                cand_rec = s23_records[cid]
                c_core = cand_rec[4]
                c_ac = cand_rec[6]
                
                # Fast pre-screening: if both name and address have negligible similarity, pair cannot match
                if s1_core and c_core and s1_ac and c_ac:
                    sim1 = fuzz.token_sort_ratio(s1_core, c_core)
                    if sim1 < 25:
                        sim2 = fuzz.token_sort_ratio(s1_ac, c_ac)
                        if sim2 < 25:
                            continue
                            
                feat = extract_pair_features(s1_rec, cand_rec)
                all_pair_feats.append(feat)
                all_pair_info.append((s1_id, cid))
                
        print(f"Retrieved candidates and extracted {len(all_pair_feats)} plausible pairs in {time.time()-t_cand:.2f}s")
        
        # Score pairs with XGBoost in streaming batches
        if len(all_pair_feats) > 0:
            X_eval = np.array(all_pair_feats, dtype=np.float32)
            probs = clf.predict_proba(X_eval)[:, 1]
            
            # Group candidate scores and apply 1-to-1 competitive matching
            cand_to_best_s1 = {} # cid -> (best_prob, s1_id)
            s1_cand_probs = defaultdict(dict) # s1_id -> {cid: prob}
            
            for (s1_id, cid), prob in zip(all_pair_info, probs):
                if prob >= threshold:
                    s1_cand_probs[s1_id][cid] = prob
                    if cid not in cand_to_best_s1 or prob > cand_to_best_s1[cid][0]:
                        cand_to_best_s1[cid] = (prob, s1_id)
                        
            # Assign match only if this S1 is the best candidate for this S2/S3 entity
            for s1_id in s1_by_country[ctry]:
                matched = []
                for cid, prob in s1_cand_probs.get(s1_id, {}).items():
                    if cand_to_best_s1.get(cid, (None, None))[1] == s1_id:
                        matched.append(cid)
                final_matches[s1_id] = matched
        else:
            for s1_id in s1_by_country[ctry]:
                final_matches[s1_id] = []
                
        # Clear country data from memory
        del inv_index
        del s23_records
        del s1_records_by_country[ctry]
        print(f"Completed {ctry} in {time.time()-t_c:.2f}s")
        
    # 4. Write output files in original order of test_source1.tsv
    print("\nWriting output TSV files...")
    
    # Write candidate_pairs.tsv
    with open(candidate_out_path, 'w', encoding='utf-8') as f_cand:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1_all_ids:
            cands = final_candidates.get(s1_id, [])
            f_cand.write(f"{s1_id}\t{','.join(cands)}\n")
            
    # Write matching_results.tsv
    with open(matching_out_path, 'w', encoding='utf-8') as f_match:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1_all_ids:
            matches = final_matches.get(s1_id, [])
            f_match.write(f"{s1_id}\t{','.join(matches)}\n")
            
    print(f"Successfully generated:")
    print(f"  {candidate_out_path}")
    print(f"  {matching_out_path}")
    print(f"Total pipeline execution time: {time.time()-t_start:.2f}s")
