"""
Analytics and Fatigue/Strain Modeling Modules.
"""

from oculisense.analytics.strain_analyzer import StrainAnalyzer, StrainResult
from oculisense.analytics.fatigue_model import FatigueModel, FatigueResult
from oculisense.analytics.break_advisor import BreakAdvisor, BreakStatus

__all__ = [
    "StrainAnalyzer",
    "StrainResult",
    "FatigueModel",
    "FatigueResult",
    "BreakAdvisor",
    "BreakStatus",
]
