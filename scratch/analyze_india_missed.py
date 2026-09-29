import sys
from collections import defaultdict, Counter

sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, 'code/business_entity_resolution')
from src.pipeline import prepare_record
from src.blocking import get_blocking_keys, retrieve_top_candidates
from src.config import TOP_CANDIDATES_PER_S1

print("Analyzing India missed matches...")

# Load 500 India S1
s1_records = {}
with open('dataset/train/train_source1.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip('\n').split('\t')
        if len(p) > 3 and p[3] == 'India':
            s1_records[p[0]] = prepare_record(p)
            if len(s1_records) >= 500: break

s1_ids = set(s1_records.keys())
gt = {}
with open('dataset/train/train_ground_truth.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip('\n').split('\t')
        if p[0] in s1_ids:
            gt[p[0]] = [x.strip() for x in p[1].split(',') if x.strip()] if len(p) > 1 and p[1].strip() else []

target_ids = {m for mlist in gt.values() for m in mlist}

# Load S2 and S3 for these targets
s23_records = {}
inv_index = defaultdict(list)
for fn in ['dataset/train/train_source2.tsv', 'dataset/train/train_source3.tsv']:
    with open(fn, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.rstrip('\n').split('\t')
            if p[0] in target_ids:
                rec = prepare_record(p)
                s23_records[p[0]] = rec
                for k in get_blocking_keys(rec[0], rec[1], rec[2]):
                    inv_index[k].append(p[0])

# Find missed pairs
missed = []
for s1_id, s1_rec in s1_records.items():
    cands = retrieve_top_candidates(s1_rec[0], s1_rec[1], 'India', inv_index, top_k=TOP_CANDIDATES_PER_S1)
    cand_set = set(cands)
    for tid in gt.get(s1_id, []):
        if tid in s23_records and tid not in cand_set:
            trec = s23_records[tid]
            # Check key overlap
            s1_keys = get_blocking_keys(s1_rec[0], s1_rec[1], 'India')
            t_keys = get_blocking_keys(trec[0], trec[1], 'India')
            overlap = s1_keys & t_keys
            missed.append((s1_rec, trec, overlap, s1_keys, t_keys))

print(f"Total targets: {len(target_ids)}, Missed: {len(missed)}")
print("\nSample Missed Matches (India):")
for s1_r, tr, ov, s1k, tk in missed[:15]:
    print(f"S1: Name='{s1_r[0]}' | Addr='{s1_r[1]}'")
    print(f"TR: Name='{tr[0]}' | Addr='{tr[1]}'")
    print(f"Shared keys ({len(ov)}): {ov}")
    if not ov:
        print(f"  S1 keys: {list(s1k)[:5]}")
        print(f"  TR keys: {list(tk)[:5]}")
    print("-" * 50)
