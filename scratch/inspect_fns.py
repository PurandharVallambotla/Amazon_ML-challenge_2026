import sys
from collections import defaultdict
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')
sys.path.insert(0, 'code/business_entity_resolution')

from scratch.test_enhanced_model import (
    extract_comprehensive_features, prepare_enhanced_record
)
from scratch.test_enhanced_blocking import (
    get_enhanced_blocking_keys, KEY_WEIGHTS, MAX_POSTINGS
)
from collections import Counter
import xgboost as xgb

# 1. Load 500 S1 records
s1_dict = {}
with open('dataset/train/train_source1.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for i, line in enumerate(f):
        if i >= 10000 and i < 11000:
            p = line.rstrip('\n').split('\t')
            s1_dict[p[0]] = prepare_enhanced_record(p)
        elif i >= 11000: break

s1_ids = set(s1_dict.keys())
gt = {}
with open('dataset/train/train_ground_truth.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip('\n').split('\t')
        if p[0] in s1_ids:
            gt[p[0]] = [x.strip() for x in p[1].split(',') if x.strip()] if len(p) > 1 and p[1].strip() else []

targets = {m for mlist in gt.values() for m in mlist}

# Load S2 and S3 for these targets
s23_dict = {}
for fn in ['dataset/train/train_source2.tsv', 'dataset/train/train_source3.tsv']:
    with open(fn, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.rstrip('\n').split('\t')
            if p[0] in targets:
                s23_dict[p[0]] = prepare_enhanced_record(p)

# Let's inspect true matches features
print("Feature analysis of true matches:")
low_core_sim = []
empty_addr_matches = []

for s1_id, tids in gt.items():
    s1_rec = s1_dict[s1_id]
    for tid in tids:
        if tid in s23_dict:
            trec = s23_dict[tid]
            feat = extract_comprehensive_features(s1_rec, trec)
            # feat: [core_sim, core_exact, core_partial, core_set, b_sim, b_exact, b_conflict, house_match, house_conflict, has_num_c, num_conf, num_sub, pcode_status, state_status, addr_sim, addr_partial, comb_sim, has_s1_addr, has_c_addr, addr_match_brand_conf, brand_match_house_conf, perfect_align]
            core_sim = feat[0]
            b_sim = feat[4]
            addr_sim = feat[14]
            has_c_addr = feat[18]
            
            if has_c_addr == 0.0:
                empty_addr_matches.append((s1_rec, trec, feat))
            elif core_sim < 0.60:
                low_core_sim.append((s1_rec, trec, feat))

print(f"Total true pairs: {sum(len(v) for v in gt.values())}")
print(f"True pairs with empty candidate address: {len(empty_addr_matches)}")
print(f"True pairs with core_sim < 0.60: {len(low_core_sim)}")

print("\n--- Sample True Pairs with Empty Candidate Address ---")
for s1_r, tr, f in empty_addr_matches[:6]:
    print(f"S1: Name='{s1_r[0]}' | Addr='{s1_r[1]}'")
    print(f"TR: Name='{tr[0]}' | Addr='{tr[1]}'")
    print(f"Feat: core_sim={f[0]:.2f}, b_sim={f[4]:.2f}")
    print()

print("\n--- Sample True Pairs with Low Core Name Similarity ---")
for s1_r, tr, f in low_core_sim[:6]:
    print(f"S1: Name='{s1_r[0]}' | Addr='{s1_r[1]}'")
    print(f"TR: Name='{tr[0]}' | Addr='{tr[1]}'")
    print(f"Feat: core_sim={f[0]:.2f}, b_sim={f[4]:.2f}, addr_sim={f[14]:.2f}, house_match={f[7]}")
    print()
