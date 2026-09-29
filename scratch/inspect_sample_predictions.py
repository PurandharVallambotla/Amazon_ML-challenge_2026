import sys
from collections import defaultdict

# Load some test S1 records
test_s1 = {}
with open('dataset/test/test_source1.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for i, line in enumerate(f):
        if i >= 100: break
        p = line.rstrip('\n').split('\t')
        test_s1[p[0]] = p

test_s1_ids = set(test_s1.keys())

# Load matching results for these
matches = {}
with open('output/matching_results.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip('\n').split('\t')
        if p[0] in test_s1_ids:
            matches[p[0]] = [x.strip() for x in p[1].split(',') if x.strip()] if len(p) > 1 and p[1].strip() else []

# Load candidate pairs for these
candidates = {}
with open('output/candidate_pairs.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip('\n').split('\t')
        if p[0] in test_s1_ids:
            candidates[p[0]] = [x.strip() for x in p[1].split(',') if x.strip()] if len(p) > 1 and p[1].strip() else []

# Find all matched/candidate IDs to look up in S2 and S3
target_ids = set()
for mlist in matches.values(): target_ids.update(mlist)
for clist in candidates.values(): target_ids.update(clist)

test_s23 = {}
for fn in ['dataset/test/test_source2.tsv', 'dataset/test/test_source3.tsv']:
    with open(fn, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.rstrip('\n').split('\t')
            if p[0] in target_ids:
                test_s23[p[0]] = p

print("=== SAMPLE INSPECTION OF PREDICTIONS ===")
for s1_id in list(test_s1.keys())[:10]:
    s1_row = test_s1[s1_id]
    s1_name, s1_addr, s1_ctry = s1_row[1], s1_row[2], s1_row[3]
    m_list = matches.get(s1_id, [])
    c_list = candidates.get(s1_id, [])
    print(f"\nS1: [{s1_id}] ({s1_ctry}) Name: '{s1_name}' | Addr: '{s1_addr}'")
    print(f"Candidates ({len(c_list)}): {c_list}")
    print(f"Matched ({len(m_list)}): {m_list}")
    for cid in m_list:
        crow = test_s23.get(cid, [cid, '???', '???'])
        print(f"   -> Match [{cid}] Name: '{crow[1]}' | Addr: '{crow[2]}'")
