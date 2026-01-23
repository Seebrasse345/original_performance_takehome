"""
Shared configuration for diagnostic scripts.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from problem import SLOT_LIMITS, VLEN, N_CORES, SCRATCH_SIZE, HASH_STAGES

# Default test configuration
DEFAULT_CONFIG = {
    "forest_height": 10,
    "n_nodes": 2047,
    "batch_size": 256,
    "rounds": 16,
    "seed": 123,
}

# Re-export for convenience
__all__ = ["DEFAULT_CONFIG", "SLOT_LIMITS", "VLEN", "N_CORES", "SCRATCH_SIZE", "HASH_STAGES"]
