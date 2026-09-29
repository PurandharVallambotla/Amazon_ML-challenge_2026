import sys
from collections import defaultdict
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'code/business_entity_resolution')
from src.pipeline import prepare_record
from src.blocking import get_blocking_keys, retrieve_top_candidates
from src.features import extract_pair_features
from src.model import load_trained_model
from src.config import MODEL_PATH, MATCH_PROB_THRESHOLD, TOP_CANDIDATES_PER_S1

# Load 1,000 US S1 records
val_s1 = {}
with open('dataset/train/train_source1.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for i, line in enumerate(f):
        if i < 200000: continue
        p = line.rstrip('\n').split('\t')
        if len(p) > 3 and p[3] == 'US':
            val_s1[p[0]] = prepare_record(p)
            if len(val_s1) >= 1000: break

val_s1_ids = set(val_s1.keys())
val_gt = {}
with open('dataset/train/train_ground_truth.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip('\n').split('\t')
        if p[0] in val_s1_ids:
            val_gt[p[0]] = [x.strip() for x in p[1].split(',') if x.strip()] if len(p) > 1 and p[1].strip() else []

# Index 1,000,000 US S2/S3
inv_index = defaultdict(list)
s23_records = {}
for fn in ['dataset/train/train_source2.tsv', 'dataset/train/train_source3.tsv']:
    with open(fn, 'r', encoding='utf-8') as f:
        next(f)
        cnt = 0
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) > 3 and p[3] == 'US':
                eid = p[0]
                rec = prepare_record(p)
                s23_records[eid] = rec
                for k in get_blocking_keys(rec[0], rec[1], rec[2]):
                    inv_index[k].append(eid)
                cnt += 1
                if cnt >= 500000: break

clf = load_trained_model(MODEL_PATH)
fps = []
for s1_id, s1_rec in val_s1.items():
    cands = retrieve_top_candidates(s1_rec[0], s1_rec[1], 'US', inv_index, top_k=TOP_CANDIDATES_PER_S1)
    gt_set = set(val_gt.get(s1_id, []))
    for cid in cands:
        if cid in s23_records:
            cand_rec = s23_records[cid]
            feat = extract_pair_features(s1_rec, cand_rec)
            prob = clf.predict_proba(np.array([feat], dtype=np.float32))[0, 1]
            if prob >= 0.78 and cid not in gt_set:
                fps.append((s1_rec, cand_rec, prob, feat))

print(f"Total false positives found: {len(fps)}")
print("\nSample False Positives (US):")
for s1_r, cr, pr, ft in fps[:15]:
    print(f"S1:   Name='{s1_r[0]}' | Addr='{s1_r[1]}'")
    print(f"Cand: Name='{cr[0]}' | Addr='{cr[1]}'")
    print(f"Prob: {pr:.4f} | Features: name_sim={ft[0]:.2f}, core_sim={ft[1]:.2f}, addr_sim={ft[4]:.2f}, num_conflict={ft[8]}")
    print("-" * 60)
