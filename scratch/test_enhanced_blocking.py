import sys
import time
from collections import defaultdict, Counter
import re

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')
sys.path.insert(0, 'code/business_entity_resolution')
from scratch.test_enhanced_features import (
    clean_name, clean_core_name, clean_addr, extract_all_numbers,
    extract_postal_code, LEGAL_SUFFIXES, COMMON_ADDR_WORDS
)

def get_enhanced_blocking_keys(name: str, addr: str, country: str):
    keys = []
    
    # --- NAME KEYS ---
    cn = clean_name(name)
    words = [w for w in cn.split() if w not in LEGAL_SUFFIXES]
    
    # 1. Full normalized name (min 4 chars)
    nn = ''.join(cn.split())
    if len(nn) >= 4:
        keys.append(f"nn:{nn}")
        
    # 2. Core words concatenated (min 4 chars)
    ncc = ''.join(words)
    if len(ncc) >= 4:
        keys.append(f"ncc:{ncc}")
        
    # 3. Individual significant words (min 4 chars, skip purely numeric)
    for w in words:
        if len(w) >= 4 and not w.isdigit():
            keys.append(f"nw:{w}")
            
    # 4. First two significant words
    if len(words) >= 2:
        keys.append(f"np:{words[0]}_{words[1]}")
        # Also second and third if available
        if len(words) >= 3:
            keys.append(f"np:{words[1]}_{words[2]}")
            
    # --- ADDRESS KEYS ---
    ac = clean_addr(addr)
    awords = ac.split()
    nums = extract_all_numbers(addr)
    # Filter numbers: keep up to 4 significant numbers
    sig_nums = [n for n in nums if len(n) <= 6][:4]
    
    sig_awords = [w for w in awords if not w.isdigit() and w not in COMMON_ADDR_WORDS and len(w) >= 4][:5]
    
    # 5. Number + street word combination
    for n in sig_nums:
        for sw in sig_awords:
            keys.append(f"ns:{n}_{sw}")
            
    # 6. Postal code + first core word or street word
    pcode = extract_postal_code(addr, country)
    if pcode:
        if words:
            keys.append(f"pc_w:{pcode}_{words[0]}")
        for sw in sig_awords[:2]:
            keys.append(f"pc_s:{pcode}_{sw}")
            
    # 7. Rare address locality words (min 6 chars)
    for aw in sig_awords:
        if len(aw) >= 6:
            keys.append(f"aw:{aw}")
            
    return {f"{country}:{k}" for k in keys}

# Let's test blocking recall on 1,000 India S1 records against 500,000 S2/S3
print("Testing enhanced blocking keys on India records...")

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
print(f"1000 India S1 have {len(target_ids)} true targets.")

# Index 500,000 India S2/S3 records
inv_index = defaultdict(list)
targets_in_index = 0
for fn in ['dataset/train/train_source2.tsv', 'dataset/train/train_source3.tsv']:
    with open(fn, 'r', encoding='utf-8') as f:
        next(f)
        cnt = 0
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) > 3 and p[3] == 'India':
                eid = p[0]
                if eid in target_ids: targets_in_index += 1
                for k in get_enhanced_blocking_keys(p[1], p[2], 'India'):
                    inv_index[k].append(eid)
                cnt += 1
                if cnt >= 250000: break

print(f"Indexed records. Targets present in index: {targets_in_index}/{len(target_ids)}")

KEY_WEIGHTS = {
    'nn': 15,
    'ncc': 12,
    'np': 10,
    'pc_w': 8,
    'ns': 8,
    'nw': 4,
    'pc_s': 4,
    'aw': 2
}

MAX_POSTINGS = {
    'nn': 1000,
    'ncc': 800,
    'np': 500,
    'pc_w': 500,
    'ns': 400,
    'nw': 200,
    'pc_s': 200,
    'aw': 100
}

for top_k in [15, 25, 35, 45]:
    found = 0
    for s1_id, (name, addr, ctry) in s1_records.items():
        cands = Counter()
        for k in get_enhanced_blocking_keys(name, addr, ctry):
            k_type = k.split(':')[1] if ':' in k else ''
            max_p = MAX_POSTINGS.get(k_type, 150)
            postings = inv_index.get(k)
            if postings and len(postings) <= max_p:
                w = KEY_WEIGHTS.get(k_type, 1)
                for cid in postings:
                    cands[cid] += w
        top_c = [c for c, _ in cands.most_common(top_k)]
        t_set = set(gt.get(s1_id, []))
        found += len(t_set & set(top_c))
    rec = found / targets_in_index if targets_in_index > 0 else 0
    print(f"Enhanced Blocking: top_k={top_k:2d} -> Recall: {found}/{targets_in_index} ({rec*100:.2f}%)")
