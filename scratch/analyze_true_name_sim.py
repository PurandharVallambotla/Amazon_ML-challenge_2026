import sys
sys.path.insert(0, 'code/business_entity_resolution')
from src.utils import clean_name, clean_core_name, clean_addr, extract_nums
from rapidfuzz import fuzz

# Let's inspect 50 true matches from train_ground_truth.tsv
s1_records = {}
with open('dataset/train/train_source1.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for i, line in enumerate(f):
        if i >= 500: break
        p = line.rstrip('\n').split('\t')
        s1_records[p[0]] = p

gt = {}
with open('dataset/train/train_ground_truth.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip('\n').split('\t')
        if p[0] in s1_records:
            gt[p[0]] = [x.strip() for x in p[1].split(',') if x.strip()] if len(p) > 1 and p[1].strip() else []

all_targets = {m for mlist in gt.values() for m in mlist}
s23_records = {}
for fn in ['dataset/train/train_source2.tsv', 'dataset/train/train_source3.tsv']:
    with open(fn, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.rstrip('\n').split('\t')
            if p[0] in all_targets:
                s23_records[p[0]] = p

print(f"Total S1 checked: {len(s1_records)}, with targets: {len(all_targets)}")

# Look at distribution of core_name_sim on true matches
name_sims = []
core_sims = []
for s1_id, targets in gt.items():
    s1_row = s1_records[s1_id]
    s1_name, s1_addr = s1_row[1], s1_row[2]
    s1_core = clean_core_name(s1_name)
    for tid in targets:
        if tid in s23_records:
            t_row = s23_records[tid]
            t_name, t_addr = t_row[1], t_row[2]
            t_core = clean_core_name(t_name)
            csim = fuzz.token_sort_ratio(s1_core, t_core)
            core_sims.append(csim)

print(f"True matches core_name_sim:")
print(f"  Min: {min(core_sims):.1f}")
print(f"  10th percentile: {sorted(core_sims)[len(core_sims)//10]:.1f}")
print(f"  Median: {sorted(core_sims)[len(core_sims)//2]:.1f}")
print(f"  Count < 60: {sum(1 for s in core_sims if s < 60)} / {len(core_sims)} ({sum(1 for s in core_sims if s < 60)/len(core_sims)*100:.2f}%)")
print(f"  Count < 70: {sum(1 for s in core_sims if s < 70)} / {len(core_sims)} ({sum(1 for s in core_sims if s < 70)/len(core_sims)*100:.2f}%)")
print(f"  Count < 80: {sum(1 for s in core_sims if s < 80)} / {len(core_sims)} ({sum(1 for s in core_sims if s < 80)/len(core_sims)*100:.2f}%)")

# Print examples where core_name_sim is low
print("\nExamples of true matches with lowest core_name_sim:")
pairs_with_sim = []
for s1_id, targets in gt.items():
    s1_row = s1_records[s1_id]
    s1_core = clean_core_name(s1_row[1])
    for tid in targets:
        if tid in s23_records:
            t_row = s23_records[tid]
            t_core = clean_core_name(t_row[1])
            csim = fuzz.token_sort_ratio(s1_core, t_core)
            pairs_with_sim.append((csim, s1_row[1], t_row[1], s1_row[2], t_row[2]))

pairs_with_sim.sort(key=lambda x: x[0])
for csim, s1n, tn, s1a, ta in pairs_with_sim[:10]:
    print(f"  Sim {csim:.1f}: '{s1n}' <-> '{tn}'")
    print(f"       Addr: '{s1a}' <-> '{ta}'")
