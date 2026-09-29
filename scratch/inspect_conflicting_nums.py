import sys
sys.path.insert(0, 'code/business_entity_resolution')
from src.utils import extract_nums

s1_rows = {}
with open('dataset/train/train_source1.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for i, line in enumerate(f):
        if i >= 1000: break
        p = line.rstrip('\n').split('\t')
        s1_rows[p[0]] = p

gt = {}
with open('dataset/train/train_ground_truth.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip('\n').split('\t')
        if p[0] in s1_rows:
            gt[p[0]] = [x.strip() for x in p[1].split(',') if x.strip()] if len(p) > 1 and p[1].strip() else []

targets = {m for mlist in gt.values() for m in mlist}
s23_rows = {}
for fn in ['dataset/train/train_source2.tsv', 'dataset/train/train_source3.tsv']:
    with open(fn, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.rstrip('\n').split('\t')
            if p[0] in targets:
                s23_rows[p[0]] = p

print("Examples of true matches where extract_nums gave disjoint sets:")
cnt = 0
for s1_id, tids in gt.items():
    s1_addr = s1_rows[s1_id][2]
    s1_nums = extract_nums(s1_addr)
    for tid in tids:
        if tid in s23_rows:
            t_addr = s23_rows[tid][2]
            t_nums = extract_nums(t_addr)
            if s1_nums and t_nums and not (s1_nums & t_nums):
                print(f"S1: {s1_rows[s1_id][1]!r} | Addr: {s1_addr!r} | Nums: {s1_nums}")
                print(f"TR: {s23_rows[tid][1]!r} | Addr: {t_addr!r} | Nums: {t_nums}")
                print()
                cnt += 1
                if cnt >= 8: break
    if cnt >= 8: break
