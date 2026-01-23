"""
Diagnostic toolkit for VLIW SIMD kernel optimization.

This package provides comprehensive analysis tools for:
- Bottleneck identification
- Operation breakdown by type/depth/round
- Scheduling quality analysis
- Memory/scratch usage analysis
- Cycle-by-cycle tracing
- A/B comparison testing
"""

from .config import DEFAULT_CONFIG, SLOT_LIMITS, VLEN
