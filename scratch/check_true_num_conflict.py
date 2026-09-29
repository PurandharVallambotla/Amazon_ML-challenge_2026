import sys
sys.path.insert(0, 'code/business_entity_resolution')
from src.utils import extract_nums

# Check how often true matches have conflicting house numbers
s1_rows = {}
with open('dataset/train/train_source1.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for i, line in enumerate(f):
        if i >= 2000: break
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

total_pairs_with_nums = 0
conflicting_pairs = 0
shared_pairs = 0

for s1_id, tids in gt.items():
    s1_addr = s1_rows[s1_id][2]
    s1_nums = extract_nums(s1_addr)
    for tid in tids:
        if tid in s23_rows:
            t_addr = s23_rows[tid][2]
            t_nums = extract_nums(t_addr)
            if s1_nums and t_nums:
                total_pairs_with_nums += 1
                if s1_nums & t_nums:
                    shared_pairs += 1
                else:
                    conflicting_pairs += 1

print(f"Total true pairs where both have numbers: {total_pairs_with_nums}")
print(f"Shared numbers: {shared_pairs} ({shared_pairs/total_pairs_with_nums*100:.2f}%)")
print(f"Conflicting numbers: {conflicting_pairs} ({conflicting_pairs/total_pairs_with_nums*100:.2f}%)")
