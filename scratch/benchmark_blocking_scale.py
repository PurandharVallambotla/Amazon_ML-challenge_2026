import sys
import time
from collections import defaultdict, Counter

sys.path.insert(0, 'code/business_entity_resolution')
from src.pipeline import prepare_record
from src.blocking import get_blocking_keys, retrieve_top_candidates
from src.config import TOP_CANDIDATES_PER_S1, MAX_POSTINGS_PER_KEY

print("Loading 2,000 US S1 records from train...")
s1_records = {}
with open('dataset/train/train_source1.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip('\n').split('\t')
        if len(p) > 3 and p[3] == 'US':
            s1_records[p[0]] = prepare_record(p)
            if len(s1_records) >= 2000:
                break

s1_ids = set(s1_records.keys())
gt = {}
with open('dataset/train/train_ground_truth.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip('\n').split('\t')
        if p[0] in s1_ids:
            gt[p[0]] = [x.strip() for x in p[1].split(',') if x.strip()] if len(p) > 1 and p[1].strip() else []

target_ids = {m for mlist in gt.values() for m in mlist}
print(f"2000 US S1 records have {len(target_ids)} true targets.")

# Now let's index 1,000,000 US S2/S3 records (including the target IDs)
print("Indexing 1,000,000 US S2 and S3 records...")
t0 = time.time()
inv_index = defaultdict(list)
indexed_targets = 0

for fn in ['dataset/train/train_source2.tsv', 'dataset/train/train_source3.tsv']:
    with open(fn, 'r', encoding='utf-8') as f:
        next(f)
        cnt = 0
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) > 3 and p[3] == 'US':
                eid = p[0]
                rec = prepare_record(p)
                if eid in target_ids:
                    indexed_targets += 1
                for k in get_blocking_keys(rec[0], rec[1], rec[2]):
                    inv_index[k].append(eid)
                cnt += 1
                if cnt >= 500000:
                    break

print(f"Indexed records in {time.time()-t0:.2f}s. Targets present in index: {indexed_targets}/{len(target_ids)}")

# Check key size distribution
key_lens = [len(v) for v in inv_index.values()]
print(f"Total keys: {len(inv_index)}")
print(f"Keys with len > 150: {sum(1 for l in key_lens if l > 150)} ({sum(1 for l in key_lens if l > 150)/len(key_lens)*100:.2f}%)")
print(f"Keys with len > 500: {sum(1 for l in key_lens if l > 500)}")

# Check recall under different max_postings and top_k
for max_postings in [150, 300, 500, 1000]:
    for top_k in [15, 25, 40]:
        t_found = 0
        for s1_id, s1_rec in s1_records.items():
            cands = Counter()
            for key in get_blocking_keys(s1_rec[0], s1_rec[1], s1_rec[2]):
                postings = inv_index.get(key)
                if postings and len(postings) <= max_postings:
                    k_type = key.split(':')[1] if ':' in key else ''
                    w = 10 if k_type in ('nn', 'ncc') else (8 if k_type == 'ns' else 2)
                    for cid in postings:
                        cands[cid] += w
            top_c = [c for c, _ in cands.most_common(top_k)]
            t_set = set(gt.get(s1_id, []))
            t_found += len(t_set & set(top_c))
        recall = t_found / indexed_targets if indexed_targets > 0 else 0
        print(f"max_postings={max_postings:4d}, top_k={top_k:2d} -> Recall: {t_found}/{indexed_targets} ({recall*100:.2f}%)")
