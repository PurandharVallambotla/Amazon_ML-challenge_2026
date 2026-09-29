import sys
import math
from collections import defaultdict, Counter

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')
sys.path.insert(0, 'code/business_entity_resolution')

from scratch.test_enhanced_features import (
    clean_name, clean_core_name, clean_addr, extract_all_numbers,
    extract_postal_code, LEGAL_SUFFIXES, COMMON_ADDR_WORDS
)
from scratch.test_enhanced_blocking import get_enhanced_blocking_keys

# Load 2,000 S1 records
s1_records = {}
with open('dataset/train/train_source1.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for i, line in enumerate(f):
        if i >= 2000: break
        p = line.rstrip('\n').split('\t')
        s1_records[p[0]] = (p[1], p[2], p[3])

s1_ids = set(s1_records.keys())
gt = {}
with open('dataset/train/train_ground_truth.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip('\n').split('\t')
        if p[0] in s1_ids:
            gt[p[0]] = [x.strip() for x in p[1].split(',') if x.strip()] if len(p) > 1 and p[1].strip() else []

targets = {m for mlist in gt.values() for m in mlist}

# Index 500,000 S2/S3 records
inv_index = defaultdict(list)
targets_present = 0
for fn in ['dataset/train/train_source2.tsv', 'dataset/train/train_source3.tsv']:
    with open(fn, 'r', encoding='utf-8') as f:
        next(f)
        cnt = 0
        for line in f:
            p = line.rstrip('\n').split('\t')
            eid = p[0]
            if eid in targets: targets_present += 1
            for k in get_enhanced_blocking_keys(p[1], p[2], p[3] if len(p) > 3 else 'Unknown'):
                inv_index[k].append(eid)
            cnt += 1
            if cnt >= 250000: break

print(f"Total targets present: {targets_present}/{len(targets)}")

# Compare Hard Cutoff vs IDF Weighting
for mode in ['hard_cutoff', 'idf_weighted']:
    for top_k in [25, 40, 50]:
        found = 0
        for s1_id, (name, addr, ctry) in s1_records.items():
            cands = Counter()
            keys = get_enhanced_blocking_keys(name, addr, ctry)
            for k in keys:
                postings = inv_index.get(k)
                if not postings: continue
                n_p = len(postings)
                if mode == 'hard_cutoff':
                    if n_p <= 200:
                        k_type = k.split(':')[1] if ':' in k else ''
                        w = 15 if k_type in ('nn', 'ncc') else (10 if k_type == 'np' else 3)
                        for cid in postings: cands[cid] += w
                else: # idf_weighted with higher cutoff
                    if n_p <= 2000:
                        k_type = k.split(':')[1] if ':' in k else ''
                        base_w = 20 if k_type in ('nn', 'ncc') else (12 if k_type == 'np' else (8 if k_type == 'ns' else 4))
                        idf = base_w / math.log(2 + n_p)
                        for cid in postings: cands[cid] += idf
            top_c = [c for c, _ in cands.most_common(top_k)]
            t_set = set(gt.get(s1_id, []))
            found += len(t_set & set(top_c))
        rec = found / targets_present if targets_present > 0 else 0
        print(f"Mode: {mode:12s} | top_k={top_k:2d} -> Recall: {found}/{targets_present} ({rec*100:.2f}%)")
