import sys
import os
from collections import defaultdict
import numpy as np

sys.path.insert(0, 'code/business_entity_resolution')
from src.pipeline import prepare_record
from src.blocking import get_blocking_keys, retrieve_top_candidates
from src.features import extract_pair_features

# Let's inspect what candidates are retrieved during train_pipeline
s1_sample = {}
with open('dataset/train/train_source1.tsv', 'r', encoding='utf-8') as fp:
    next(fp)
    for i, line in enumerate(fp):
        if i >= 5000: break
        p = line.rstrip('\n').split('\t')
        s1_sample[p[0]] = prepare_record(p)

gt_map = {}
with open('dataset/train/train_ground_truth.tsv', 'r', encoding='utf-8') as fp:
    next(fp)
    for line in fp:
        p = line.rstrip('\n').split('\t')
        if p[0] in s1_sample:
            gt_map[p[0]] = [x.strip() for x in p[1].split(',') if x.strip()] if len(p) > 1 and p[1].strip() else []

all_targets = {m for mlist in gt_map.values() for m in mlist}

# Load first 20k distractors
s23_records = {}
inv_index = defaultdict(list)
for fn in ['dataset/train/train_source2.tsv', 'dataset/train/train_source3.tsv']:
    with open(fn, 'r', encoding='utf-8') as fp:
        next(fp)
        cnt = 0
        for line in fp:
            p = line.rstrip('\n').split('\t')
            eid = p[0]
            if eid in all_targets:
                rec = prepare_record(p)
                s23_records[eid] = rec
                for k in get_blocking_keys(rec[0], rec[1], rec[2]):
                    inv_index[k].append(eid)
            elif cnt < 10000:
                rec = prepare_record(p)
                s23_records[eid] = rec
                for k in get_blocking_keys(rec[0], rec[1], rec[2]):
                    inv_index[k].append(eid)
                cnt += 1

print(f"Indexed {len(s23_records)} records.")

neg_pairs = []
for s1_id, s1_rec in list(s1_sample.items())[:1000]:
    cands = retrieve_top_candidates(s1_rec[0], s1_rec[1], s1_rec[2], inv_index, top_k=20)
    gt_set = set(gt_map.get(s1_id, []))
    for cid in cands:
        if cid not in gt_set and cid in s23_records:
            cand_rec = s23_records[cid]
            feat = extract_pair_features(s1_rec, cand_rec)
            neg_pairs.append((s1_rec, cand_rec, feat))

print(f"Total negative pairs in top-20 candidates for 1000 S1: {len(neg_pairs)}")
if neg_pairs:
    print("\nSample negative pairs:")
    for s1_r, c_r, f in neg_pairs[:10]:
        print(f"S1: '{s1_r[0]}' | Addr: '{s1_r[1]}'")
        print(f"Cand: '{c_r[0]}' | Addr: '{c_r[1]}'")
        print(f"Features: name_sim={f[0]:.2f}, core_sim={f[1]:.2f}, addr_sim={f[4]:.2f}, num_conflict={f[8]}")
        print()
