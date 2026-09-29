import sys
import os
import time

sys.path.insert(0, 'code/business_entity_resolution')
from src.pipeline import run_inference, prepare_record
from src.blocking import get_blocking_keys, retrieve_top_candidates
from src.features import extract_pair_features
from src.model import load_trained_model
from src.config import MODEL_PATH

clf = load_trained_model(MODEL_PATH)
print("Loaded newly trained model successfully.")

# Quick test with 1 record
r1 = prepare_record(['S1-001', 'Zephay Labs Inc', '2621 Cotten Road, Tyler, TX', 'US'])
r2 = prepare_record(['S2-001', 'Inc. Zephay Labs', '2621 Cotten Road, Tyler, Texas', 'US'])
r3 = prepare_record(['S2-002', 'Sweet Barbershop', '5131 Copper Meadow Ln, UT', 'US'])

f1 = extract_pair_features(r1, r2)
f2 = extract_pair_features(r1, r3)

p1 = clf.predict_proba([f1])[0, 1]
p2 = clf.predict_proba([f2])[0, 1]

print(f"True Match Prob: {p1:.4f} (Expected > 0.95)")
print(f"Distractor Prob: {p2:.4f} (Expected < 0.05)")

assert p1 > 0.90, f"Expected p1 > 0.90, got {p1}"
assert p2 < 0.10, f"Expected p2 < 0.10, got {p2}"
print("Model sanity check PASSED!")
