import sys
from collections import defaultdict
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')
sys.path.insert(0, 'code/business_entity_resolution')

from scratch.test_enhanced_features import clean_name, clean_core_name, clean_addr
from scratch.test_97_target import prepare_ultra_record, extract_ultra_features
from scratch.test_final_evaluation import main

# Let's inspect true targets that got prob < 0.70
print("Analyzing true targets that got probability < 0.70...")
