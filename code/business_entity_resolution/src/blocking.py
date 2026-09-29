"""Country-agnostic blocking and candidate generation using an inverted index."""

import re
import math
from collections import defaultdict, Counter
from src.config import (
    LEGAL_SUFFIXES, COMMON_ADDR_WORDS, KEY_BASE_WEIGHTS,
    MAX_POSTINGS_PER_KEY, TOP_CANDIDATES_PER_S1
)
from src.utils import (
    clean_name, clean_core_name, clean_addr, extract_all_numbers, extract_postal_code
)

def get_blocking_keys(name: str, addr: str, country: str):
    """
    Generate composite blocking keys across name and address components.
    Country is treated as an open set of string labels without hardcoding.
    """
    keys = []
    clean_ctry = country.strip().upper() if country and country.strip() else "GLOBAL"
    
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
            
    # 4. Bigrams of significant words
    if len(words) >= 2:
        keys.append(f"np:{words[0]}_{words[1]}")
        if len(words) >= 3:
            keys.append(f"np:{words[1]}_{words[2]}")
            
    # --- ADDRESS KEYS ---
    ac = clean_addr(addr)
    awords = ac.split()
    nums = extract_all_numbers(addr)
    sig_nums = [n for n in nums if len(n) <= 6][:4]
    sig_awords = [w for w in awords if not w.isdigit() and w not in COMMON_ADDR_WORDS and len(w) >= 4][:5]
    
    # 5. Number + street word combination
    for n in sig_nums:
        for sw in sig_awords:
            keys.append(f"ns:{n}_{sw}")
            
    # 6. Postal code + word combinations
    pcode = extract_postal_code(addr)
    if pcode:
        if words:
            keys.append(f"pc_w:{pcode}_{words[0]}")
        for sw in sig_awords[:2]:
            keys.append(f"pc_s:{pcode}_{sw}")
            
    # 7. Rare address locality words (min 6 chars)
    for aw in sig_awords:
        if len(aw) >= 6:
            keys.append(f"aw:{aw}")
            
    return {f"{clean_ctry}:{k}" for k in keys}

def build_inverted_index(records_dict: dict):
    """
    Build inverted index from candidate records.
    records_dict: {entity_id: (name, addr, country, ...)}
    Returns: defaultdict(list) mapping blocking_key -> list of entity_ids
    """
    inv_index = defaultdict(list)
    for eid, rec in records_dict.items():
        name, addr, country = rec[0], rec[1], rec[2]
        for key in get_blocking_keys(name, addr, country):
            inv_index[key].append(eid)
    return inv_index

def retrieve_top_candidates(name: str, addr: str, country: str, inv_index: dict,
                            top_k: int = TOP_CANDIDATES_PER_S1,
                            max_postings: int = MAX_POSTINGS_PER_KEY):
    """
    Retrieve and rank candidate entity IDs for a Source 1 entity using IDF-weighted key scores.
    Returns: list of candidate entity IDs (at most top_k).
    """
    cand_scores = Counter()
    for key in get_blocking_keys(name, addr, country):
        postings = inv_index.get(key)
        if not postings:
            continue
        n_p = len(postings)
        if n_p <= max_postings:
            k_type = key.split(':')[1] if ':' in key else ''
            base_w = KEY_BASE_WEIGHTS.get(k_type, 3.0)
            idf_weight = base_w / math.log(2 + n_p)
            for cid in postings:
                cand_scores[cid] += idf_weight
                
    if not cand_scores:
        return []
    return [c for c, _ in cand_scores.most_common(top_k)]
