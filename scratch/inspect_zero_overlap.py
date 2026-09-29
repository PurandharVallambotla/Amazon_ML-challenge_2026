import sys
from collections import defaultdict, Counter

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')
sys.path.insert(0, 'code/business_entity_resolution')
from scratch.test_enhanced_features import (
    clean_name, clean_core_name, clean_addr, extract_all_numbers,
    extract_postal_code, LEGAL_SUFFIXES, COMMON_ADDR_WORDS
)
from scratch.test_enhanced_blocking import get_enhanced_blocking_keys, KEY_WEIGHTS, MAX_POSTINGS

# Check the missed ones
s1_records = {}
with open('dataset/train/train_source1.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip('\n').split('\t')
        if len(p) > 3 and p[3] == 'India':
            s1_records[p[0]] = (p[1], p[2], p[3])
            if len(s1_records) >= 1000: break

s1_ids = set(s1_records.keys())
gt = {}
with open('dataset/train/train_ground_truth.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip('\n').split('\t')
        if p[0] in s1_ids:
            gt[p[0]] = [x.strip() for x in p[1].split(',') if x.strip()] if len(p) > 1 and p[1].strip() else []

target_ids = {m for mlist in gt.values() for m in mlist}

# Load S2 and S3 for these targets ONLY
inv_index = defaultdict(list)
s23_records = {}
for fn in ['dataset/train/train_source2.tsv', 'dataset/train/train_source3.tsv']:
    with open(fn, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.rstrip('\n').split('\t')
            if p[0] in target_ids:
                s23_records[p[0]] = (p[1], p[2], p[3])
                for k in get_enhanced_blocking_keys(p[1], p[2], 'India'):
                    inv_index[k].append(p[0])

missed = []
for s1_id, (s1_n, s1_a, s1_c) in s1_records.items():
    s1_keys = get_enhanced_blocking_keys(s1_n, s1_a, s1_c)
    for tid in gt.get(s1_id, []):
        if tid in s23_records:
            t_n, t_a, t_c = s23_records[tid]
            t_keys = get_enhanced_blocking_keys(t_n, t_a, t_c)
            if not (s1_keys & t_keys):
                missed.append((s1_n, s1_a, t_n, t_a))

print(f"Total zero-key overlap pairs out of {len(target_ids)}: {len(missed)} ({len(missed)/len(target_ids)*100:.2f}%)")
print("\nZero-key overlap pairs:")
for s1_n, s1_a, t_n, t_a in missed[:15]:
    print(f"S1: Name='{s1_n}' | Addr='{s1_a}'")
    print(f"TR: Name='{t_n}' | Addr='{t_a}'")
    print("-" * 50)
