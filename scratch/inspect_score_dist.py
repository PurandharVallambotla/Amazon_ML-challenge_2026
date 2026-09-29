import sys
from collections import Counter
import numpy as np

# Let's inspect why Macro F0.5 is 0.935
# We can run evaluation on the 5000 S1 and break down the score by category:
# - Perfect (score = 1.0)
# - Partial (0.0 < score < 1.0)
# - Zero score (score = 0.0) -> breakdown into:
#     * True singletons predicted non-empty (false merge)
#     * True matches predicted empty (missed completely)
#     * Predictions made with 0 true matches (pure FP)
