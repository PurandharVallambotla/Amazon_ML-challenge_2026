from collections import defaultdict

s1_country = {}
with open('dataset/train/train_source1.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip('\n').split('\t')
        s1_country[p[0]] = p[3] if len(p) > 3 else 'Unknown'

gt_by_ctry = defaultdict(lambda: {'total': 0, 'empty': 0, 'non_empty': 0, 'targets': 0})
with open('dataset/train/train_ground_truth.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip('\n').split('\t')
        eid = p[0]
        ctry = s1_country.get(eid, 'Unknown')
        stats = gt_by_ctry[ctry]
        stats['total'] += 1
        if len(p) <= 1 or not p[1].strip():
            stats['empty'] += 1
        else:
            stats['non_empty'] += 1
            t_list = [x for x in p[1].split(',') if x.strip()]
            stats['targets'] += len(t_list)

for ctry, stats in sorted(gt_by_ctry.items()):
    tot = stats['total']
    emp = stats['empty']
    nemp = stats['non_empty']
    targ = stats['targets']
    avg_t = targ / nemp if nemp > 0 else 0
    print(f'Train Ground Truth - Country: {ctry}')
    print(f'  Total S1: {tot}')
    print(f'  Empty: {emp} ({emp/tot*100:.2f}%)')
    print(f'  Non-empty: {nemp} ({nemp/tot*100:.2f}%)')
    print(f'  Total targets: {targ}')
    print(f'  Avg targets per non-empty: {avg_t:.2f}')
