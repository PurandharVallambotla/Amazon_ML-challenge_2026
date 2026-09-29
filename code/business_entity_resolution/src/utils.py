"""Universal text processing, normalization, and evaluation utilities.

Strictly country-agnostic and self-contained with zero external dependencies.
"""

import re
import unicodedata
from src.config import LEGAL_SUFFIXES, COMMON_ADDR_WORDS

OFFSET_TO_LATIN = {
    # Independent vowels
    0x05: 'a', 0x06: 'aa', 0x07: 'i', 0x08: 'ee', 0x09: 'u', 0x0a: 'oo', 0x0b: 'r',
    0x0d: 'e', 0x0e: 'e', 0x0f: 'e', 0x10: 'ai', 0x11: 'o', 0x12: 'o', 0x13: 'o', 0x14: 'au',
    # Consonants
    0x15: 'k', 0x16: 'kh', 0x17: 'g', 0x18: 'gh', 0x19: 'ng',
    0x1a: 'c', 0x1b: 'ch', 0x1c: 'j', 0x1d: 'jh', 0x1e: 'ny',
    0x1f: 't', 0x20: 'th', 0x21: 'd', 0x22: 'dh', 0x23: 'n',
    0x24: 't', 0x25: 'th', 0x26: 'd', 0x27: 'dh', 0x28: 'n',
    0x2a: 'p', 0x2b: 'ph', 0x2c: 'b', 0x2d: 'bh', 0x2e: 'm',
    0x2f: 'y', 0x30: 'r', 0x32: 'l', 0x33: 'l', 0x35: 'v',
    0x36: 'sh', 0x37: 'sh', 0x38: 's', 0x39: 'h',
    # Dependent vowel signs (matras)
    0x3e: 'aa', 0x3f: 'i', 0x40: 'ee', 0x41: 'u', 0x42: 'oo', 0x43: 'r',
    0x45: 'e',  # CANDRA E
    0x46: 'e', 0x47: 'e', 0x48: 'ai',
    0x49: 'o',  # CANDRA O
    0x4a: 'o', 0x4b: 'o', 0x4c: 'au',
    0x01: 'n',  # CHANDRABINDU
    0x02: 'n',  # ANUSVARA
}

def indic_to_latin(text: str) -> str:
    """Map Brahmic/Indic script characters (Devanagari, Bengali, Tamil, etc.) to Latin phonetically."""
    if not text:
        return ""
    out = []
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x0D7F:
            block_base = (cp // 0x80) * 0x80
            offset = cp - block_base
            out.append(OFFSET_TO_LATIN.get(offset, ''))
        else:
            out.append(ch)
    return ''.join(out)

def strip_accents(text: str) -> str:
    """Universal Unicode normalization: transliterates non-Latin scripts and strips combining accents."""
    if not text:
        return ""
    text = indic_to_latin(text)
    text = text.lower().replace('ii', 'ee').replace('uu', 'oo').replace('ph', 'f').replace('w', 'v')
    return ''.join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))

def clean_name(s: str) -> str:
    """Clean business name: lowercase, strip domains/handles, remove non-alphanumeric chars."""
    if not s:
        return ""
    s = strip_accents(s.lower())
    s = re.sub(r'\.(com|org|net|in|co|fr|gov|edu|de|io|ai)\b', '', s)
    s = re.sub(r'[^a-z0-9]', ' ', s)
    return ' '.join(s.split())

def clean_core_name(s: str) -> str:
    """Extract distinctive core words from business name by removing legal suffixes."""
    words = [w for w in clean_name(s).split() if w not in LEGAL_SUFFIXES and len(w) >= 2]
    return ' '.join(words)

def clean_addr(s: str) -> str:
    """Clean address: lowercase, strip accents, replace punctuation with spaces."""
    if not s:
        return ""
    s = strip_accents(s.lower())
    s = re.sub(r'[^a-z0-9]', ' ', s)
    return ' '.join(s.split())

def extract_all_numbers(s: str) -> set:
    """Extract all digit sequences from text, stripped of leading zeros."""
    if not s:
        return set()
    return {m.lstrip('0') or '0' for m in re.findall(r'\d+', s)}

def extract_primary_house_number(addr: str) -> str:
    """Extract primary house / plot / door / unit number at start or with number prefixes."""
    if not addr:
        return None
    m = re.search(r'^\s*#*\s*(\d+)', addr)
    if m:
        return m.group(1).lstrip('0') or '0'
    m = re.search(r'\b(?:no|door|plot|flat|h\.no|wz|b|d|ews|pl|ste|suite|apt|unit)\s*[:#.-]?\s*(\d+)', addr, re.IGNORECASE)
    if m:
        return m.group(1).lstrip('0') or '0'
    m = re.search(r'\b\d+\b', addr)
    if m:
        return m.group(0).lstrip('0') or '0'
    return None

def extract_postal_code(addr: str):
    """Country-agnostic postal code candidate extraction: any 4 to 8 digit number."""
    if not addr:
        return None
    matches = re.findall(r'\b\d{4,8}\b', addr)
    return matches[-1] if matches else None

def extract_geo_tokens(addr: str) -> set:
    """Universal country-agnostic geographic locality tokens (city/state/region/province)."""
    if not addr:
        return set()
    parts = [p.strip().lower() for p in re.split(r'[,]+', addr) if p.strip()]
    geo_words = set()
    for part in parts[-2:]:
        for w in re.findall(r'[a-z]{3,}', part):
            if w not in COMMON_ADDR_WORDS:
                geo_words.add(w)
    return geo_words

def compute_macro_f05(gt_map: dict, pred_map: dict) -> float:
    """
    Compute macro-average F_0.5 score per README.md:
    F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    Singletons evaluate to 1.0 when correctly predicted empty, 0.0 otherwise.
    """
    scores = []
    for s1_id, gt_list in gt_map.items():
        gt_set = set(gt_list)
        pred_set = set(pred_map.get(s1_id, []))
        if len(gt_set) == 0:
            scores.append(1.0 if len(pred_set) == 0 else 0.0)
        else:
            if len(pred_set) == 0:
                scores.append(0.0)
            else:
                tp = len(gt_set & pred_set)
                fp = len(pred_set - gt_set)
                fn = len(gt_set - pred_set)
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                denom = 0.25 * prec + rec
                scores.append((1.25 * prec * rec) / denom if denom > 0 else 0.0)
    return float(sum(scores) / len(scores)) if scores else 0.0
