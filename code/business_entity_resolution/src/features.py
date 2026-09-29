"""Feature extraction for entity candidate pairs.

Strictly country-agnostic with 26 discriminative features.
"""

import re
from rapidfuzz import fuzz, distance
from src.config import GENERIC_INDUSTRY_WORDS

FEATURE_NAMES = [
    'core_sim', 'core_exact', 'core_partial', 'core_set', 'ccore_sim', 'max_dba_sim',
    'b_sim', 'b_exact', 'b_conflict',
    'house_match', 'house_conflict',
    'has_num_c', 'num_conf', 'num_sub',
    'pcode_status', 'geo_match', 'geo_conflict',
    'addr_sim', 'addr_partial', 'addr_conflict', 'exact_name_empty_addr',
    'comb_sim', 'has_s1_addr', 'has_c_addr',
    'addr_match_brand_conf', 'brand_match_house_conf', 'perfect_align'
]

def extract_brand_words(core_name: str) -> list:
    """Extract distinctive brand tokens by removing generic industry tokens."""
    tokens = [w for w in core_name.split() if len(w) >= 2]
    brand = [w for w in tokens if w not in GENERIC_INDUSTRY_WORDS]
    return brand if brand else tokens

def check_number_compatibility(s1_nums: set, c_nums: set) -> tuple:
    """
    Check compatibility between two sets of address numbers.
    Returns (has_common, is_conflict, is_subset_or_subnum)
    """
    if not s1_nums or not c_nums:
        return (0.0, 0.0, 0.0)
    if s1_nums & c_nums:
        return (1.0, 0.0, 1.0)
    for n1 in s1_nums:
        for n2 in c_nums:
            if n1 in n2 or n2 in n1:
                return (0.5, 0.0, 0.5)
            if len(n1) >= 4 and len(n2) >= 4 and abs(len(n1) - len(n2)) <= 1:
                if distance.Levenshtein.distance(n1, n2) <= 1:
                    return (0.5, 0.0, 0.5)
    return (0.0, 1.0, 0.0)

def compute_brand_features(s1_core: str, c_core: str) -> tuple:
    s1_brand = extract_brand_words(s1_core)
    c_brand = extract_brand_words(c_core)
    s1_b_str = ' '.join(s1_brand)
    c_b_str = ' '.join(c_brand)
    
    brand_sim = fuzz.token_sort_ratio(s1_b_str, c_b_str) / 100.0 if (s1_b_str and c_b_str) else 0.0
    brand_exact = 1.0 if s1_b_str == c_b_str and s1_b_str else 0.0
    
    s1_b_set = set(s1_brand)
    c_b_set = set(c_brand)
    
    fuzzy_brand_match = 0
    for w1 in s1_b_set:
        for w2 in c_b_set:
            if fuzz.ratio(w1, w2) >= 75:
                fuzzy_brand_match += 1
                break
                
    has_brand_conflict = 0.0
    if s1_b_set and c_b_set and fuzzy_brand_match == 0:
        has_brand_conflict = 1.0
        
    return (brand_sim, brand_exact, has_brand_conflict, len(s1_b_set), len(c_b_set))

def extract_pair_features(s1_rec: tuple, c_rec: tuple) -> list:
    """
    Extract 27 discriminative country-agnostic features for an entity candidate pair.
    rec: (name, addr, ctry, nc, core, concat_core, ac, all_nums, p_num, pcode_candidates, geo_tokens, dba_cores)
    """
    _, _, _, s1_nc, s1_core, s1_ccore, s1_ac, s1_nums, s1_pnum, s1_pcodes, s1_geo, s1_dbas = s1_rec
    _, _, _, c_nc, c_core, c_ccore, c_ac, c_nums, c_pnum, c_pcodes, c_geo, c_dbas = c_rec
    
    # 1. Fast pre-check: if both names and addresses are completely dissimilar, early exit
    core_sim = fuzz.token_sort_ratio(s1_core, c_core) / 100.0 if (s1_core and c_core) else 0.0
    addr_sim = fuzz.token_sort_ratio(s1_ac, c_ac) / 100.0 if (s1_ac and c_ac) else 0.0
    
    # 2. Core Name metrics
    core_exact = 1.0 if s1_core and c_core and s1_core == c_core else 0.0
    core_partial = fuzz.partial_ratio(s1_core, c_core) / 100.0 if (s1_core and c_core) else 0.0
    core_set = fuzz.token_set_ratio(s1_core, c_core) / 100.0 if (s1_core and c_core) else 0.0
    ccore_sim = fuzz.ratio(s1_ccore, c_ccore) / 100.0 if (s1_ccore and c_ccore) else 0.0
    
    # 3. DBA / Sub-name cross matching
    max_dba_sim = core_sim
    for dp1 in s1_dbas:
        for dp2 in c_dbas:
            if dp1 and dp2:
                sim = fuzz.token_sort_ratio(dp1, dp2) / 100.0
                if sim > max_dba_sim:
                    max_dba_sim = sim
                
    # 4. Brand features
    b_sim, b_exact, b_conflict, _, _ = compute_brand_features(s1_core, c_core)
    
    # 5. Primary house number
    house_match = 0.0
    house_conflict = 0.0
    if s1_pnum and c_pnum:
        if s1_pnum == c_pnum:
            house_match = 1.0
        elif s1_pnum in c_pnum or c_pnum in s1_pnum:
            house_match = 0.5
        else:
            house_conflict = 1.0
            
    # 6. All numbers compatibility
    has_num_c, num_conf, num_sub = check_number_compatibility(s1_nums, c_nums)
    
    # 7. Postal code consistency (country-agnostic 4-8 digit numbers)
    pcode_status = 0.0
    if s1_pcodes and c_pcodes:
        if s1_pcodes & c_pcodes:
            pcode_status = 1.0
        else:
            pcode_status = -1.0
            
    # 8. Geographic locality consistency (country-agnostic city/region/province)
    geo_match = 0.5
    geo_conflict = 0.0
    if s1_geo and c_geo:
        if s1_geo & c_geo:
            geo_match = float(len(s1_geo & c_geo) / len(s1_geo | c_geo))
            geo_conflict = 0.0
        else:
            geo_match = 0.0
            geo_conflict = 1.0
            
    # 9. Address similarities
    has_s1_addr = 1.0 if s1_ac else 0.0
    has_c_addr = 1.0 if c_ac else 0.0
    addr_partial = fuzz.partial_ratio(s1_ac, c_ac) / 100.0 if (s1_ac and c_ac) else 0.0
    addr_conflict = 1.0 if has_s1_addr == 1.0 and has_c_addr == 1.0 and addr_sim < 0.25 else 0.0
    exact_name_empty_addr = 1.0 if (b_exact == 1.0 or core_exact == 1.0 or ccore_sim >= 0.95) and (has_c_addr == 0.0 or has_s1_addr == 0.0) else 0.0
    
    # 10. Combined string similarity
    s1_comb = f"{s1_core} {s1_ac}"
    c_comb = f"{c_core} {c_ac}"
    comb_sim = fuzz.token_sort_ratio(s1_comb, c_comb) / 100.0
    
    # 11. High-precision rule flags
    addr_match_brand_conf = 1.0 if addr_sim >= 0.75 and b_conflict == 1.0 else 0.0
    brand_match_house_conf = 1.0 if (b_sim >= 0.8 or core_sim >= 0.8) and house_conflict == 1.0 else 0.0
    perfect_align = 1.0 if (b_exact == 1.0 or core_exact == 1.0 or ccore_sim >= 0.95) and (house_match == 1.0 or not (s1_pnum and c_pnum)) and geo_conflict == 0.0 else 0.0

    return [
        core_sim, core_exact, core_partial, core_set, ccore_sim, max_dba_sim,
        b_sim, b_exact, b_conflict,
        house_match, house_conflict,
        has_num_c, num_conf, num_sub,
        pcode_status, geo_match, geo_conflict,
        addr_sim, addr_partial, addr_conflict, exact_name_empty_addr,
        comb_sim, has_s1_addr, has_c_addr,
        addr_match_brand_conf, brand_match_house_conf, perfect_align
    ]
