import sys
import os
import re
import unicodedata
from collections import defaultdict, Counter
import numpy as np
from rapidfuzz import fuzz, distance
import xgboost as xgb

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'code/business_entity_resolution')
from src.config import LEGAL_SUFFIXES, COMMON_ADDR_WORDS

# 1. Enhanced Brahmic Transliteration with all missing vowel signs
FULL_OFFSET_TO_LATIN = {
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
    0x45: 'e',  # CANDRA E (ॅ, bank)
    0x46: 'e', 0x47: 'e', 0x48: 'ai',
    0x49: 'o',  # CANDRA O (ॉ, logistics, hospital, doctor)
    0x4a: 'o', 0x4b: 'o', 0x4c: 'au',
    0x01: 'n',  # CHANDRABINDU
    0x02: 'n',  # ANUSVARA
}

def indic_to_latin(text: str) -> str:
    if not text: return ""
    out = []
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x0D7F:
            base = (cp // 0x80) * 0x80
            off = cp - base
            out.append(FULL_OFFSET_TO_LATIN.get(off, ''))
        else:
            out.append(ch)
    return ''.join(out)

def normalize_text(text: str) -> str:
    if not text: return ""
    text = indic_to_latin(text)
    # Common transliteration variations
    text = text.lower().replace('ii', 'ee').replace('uu', 'oo').replace('ph', 'f').replace('w', 'v')
    return ''.join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))

def clean_name(s: str) -> str:
    if not s: return ""
    s = normalize_text(s)
    s = re.sub(r'\.(com|org|net|in|co|fr|gov|edu)\b', '', s)
    s = re.sub(r'[^a-z0-9]', ' ', s)
    return ' '.join(s.split())

def clean_core_name(s: str) -> str:
    words = [w for w in clean_name(s).split() if w not in LEGAL_SUFFIXES and len(w) >= 2]
    return ' '.join(words)

def clean_addr(s: str) -> str:
    if not s: return ""
    s = normalize_text(s)
    s = re.sub(r'[^a-z0-9]', ' ', s)
    return ' '.join(s.split())

def extract_all_numbers(s: str) -> set:
    """Extract ALL digit sequences from text, stripped of leading zeros."""
    if not s: return set()
    return {m.lstrip('0') or '0' for m in re.findall(r'\d+', s)}

def extract_postal_code(addr: str, ctry: str):
    """Extract postal code: 5 digits for US/FR, 6 digits for India."""
    if not addr: return None
    if ctry == 'India':
        m = re.findall(r'\b\d{6}\b', addr)
        return m[-1] if m else None
    elif ctry in ('US', 'France'):
        m = re.findall(r'\b\d{5}\b', addr)
        return m[-1] if m else None
    return None

print("Enhanced text normalizer defined.")
