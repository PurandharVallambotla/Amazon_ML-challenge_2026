"""Configuration and constants for Business Entity Resolution pipeline.

Strictly country-agnostic and self-contained: no country-specific hardcoding,
treating country as an open set of string labels. Zero external data.
"""

import os

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "dataset")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
DEFAULT_MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "xgboost_er_model.json")

# Candidate generation parameters
MAX_POSTINGS_PER_KEY = 2500
TOP_CANDIDATES_PER_S1 = 30
MATCH_PROB_THRESHOLD = 0.78

# Universal legal entity suffixes (international corporate suffixes across languages)
LEGAL_SUFFIXES = {
    'inc', 'incorporated', 'llc', 'ltd', 'limited', 'pvt', 'private', 'corp', 'corporation',
    'co', 'company', 'the', 'and', 'of', 'in', 'at', 'on', 'for', 'dr', 'smt', 'shri',
    'mr', 'mrs', 'ms', 'sarl', 'sasu', 'sas', 'eurl', 'sa', 'services', 'enterprises', 'center',
    'group', 'holdings', 'solutions', 'partners', 'llp', 'p', 'c', 'gmbh', 'bv', 'nv',
    'technologies', 'technology', 'associates', 'industries', 'international', 'society', 'foundation',
    'management', 'consultants', 'advisors', 'institute', 'trust', 'board', 'council', 'm/s', 'dr.',
    'sci', 'snc', 'fils', 'cie', 'societe', 'association', 'syndicat', 'groupe',
    'sarviises', 'praaivet', 'praivet', 'enterprises'
}

# Generic industry keywords that dilute distinctive brand names
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

# Common address tokens to ignore for blocking key pairing
COMMON_ADDR_WORDS = {
    'st', 'street', 'rd', 'road', 'ave', 'avenue', 'blvd', 'boulevard', 'dr', 'drive',
    'ln', 'lane', 'ct', 'court', 'way', 'pkwy', 'parkway', 'cir', 'circle', 'ter', 'terrace',
    'trl', 'trail', 'pl', 'place', 'hwy', 'highway', 'suite', 'ste', 'apt', 'apartment',
    'unit', 'fl', 'floor', 'bldg', 'building', 'box', 'po', 'near', 'opp', 'opposite',
    'behind', 'beside', 'road', 'nagar', 'colony', 'block', 'sector', 'phase', 'cross',
    'main', 'layout', 'dist', 'district', 'town', 'village', 'city', 'post', 'null', 'na',
    'rue', 'bd', 'chemin', 'impasse', 'allee', 'avenue', 'place',
    'first', 'second', 'third', '1st', '2nd', '3rd', '4th', '5th'
}

# Blocking key priority base weights for IDF scoring
KEY_BASE_WEIGHTS = {
    'nn': 20.0,
    'ncc': 18.0,
    'np': 14.0,
    'pc_w': 10.0,
    'ns': 10.0,
    'nw': 5.0,
    'pc_s': 4.0,
    'aw': 2.0
}
