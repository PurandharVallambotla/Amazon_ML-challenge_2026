import sys
import re
import unicodedata
from rapidfuzz import fuzz, distance

GENERIC_INDUSTRY_WORDS = {
    'consultants', 'consultant', 'consulting', 'enterprises', 'enterprise', 'services', 'service',
    'solutions', 'solution', 'technologies', 'technology', 'tech', 'industries', 'industry',
    'holdings', 'holding', 'associates', 'associate', 'management', 'trading', 'group',
    'agency', 'system', 'systems', 'center', 'centre', 'institute', 'foundation',
    'properties', 'property', 'realty', 'realestate', 'builders', 'builder', 'developers', 'developer',
    'logistics', 'transport', 'motors', 'motor', 'auto', 'foods', 'food', 'bakery',
    'hotel', 'hotels', 'restaurant', 'restaurants', 'cafe', 'hospital', 'hospitals',
    'clinic', 'clinics', 'dental', 'pharma', 'pharmaceuticals', 'pharmacy', 'labs', 'laboratory',
    'healthcare', 'care', 'security', 'cleaning', 'barbershop', 'salon', 'jewellers',
    'textiles', 'garments', 'clothing', 'retail', 'mart', 'store', 'stores',
    'international', 'global', 'national', 'prime', 'apex', 'club', 'ecole', 'school', 'college',
    'commercial', 'corporation', 'corp', 'limited', 'ltd', 'private', 'pvt', 'inc', 'llc', 'llp'
}

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
    # Check if any number is a substring/prefix (e.g. 17 in 177, 2543 in 25432, 085 in 7085)
    for n1 in s1_nums:
        for n2 in c_nums:
            if n1 in n2 or n2 in n1:
                return (0.5, 0.0, 0.5)
            # Edit distance <= 1 on long numbers (>= 4 digits, e.g. typo 60371 vs 60372)
            if len(n1) >= 4 and len(n2) >= 4 and abs(len(n1) - len(n2)) <= 1:
                if distance.Levenshtein.distance(n1, n2) <= 1:
                    return (0.5, 0.0, 0.5)
    # Hard conflict: both have distinct, incompatible house numbers
    return (0.0, 1.0, 0.0)

def compute_brand_features(s1_core: str, c_core: str) -> tuple:
    s1_brand = extract_brand_words(s1_core)
    c_brand = extract_brand_words(c_core)
    s1_b_str = ' '.join(s1_brand)
    c_b_str = ' '.join(c_brand)
    
    brand_sim = fuzz.token_sort_ratio(s1_b_str, c_b_str) / 100.0 if (s1_b_str and c_b_str) else 0.0
    brand_exact = 1.0 if s1_b_str == c_b_str and s1_b_str else 0.0
    
    # Check for brand conflict: S1 has brand word and Cand has brand word that are completely disjoint
    s1_b_set = set(s1_brand)
    c_b_set = set(c_brand)
    common_b = s1_b_set & c_b_set
    
    # Check fuzzy overlap of brand tokens
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

print("Brand and Number compatibility logic defined.")
