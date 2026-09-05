"""
Calibrated Meta-Labeling Framework for Equity Trading Systems.
Modular quantitative implementation.
"""

from .data_integrity import Layer1DataIntegrity, Layer2MicrostructureTopologies
from .labeling import Layer0TripleBarrier
from .econometric_engine import (
    Stage2SingleAssetDGPScanner,
    Layer10FeatureRouter,
    Stage2DiagnosticEngine,
    Stage3FeatureEngine,
)
from .feature_selection import (
    Layer12RedundancyControl,
    Layer13PredictiveDiagnostics,
    Layer14LagTransformEngine,
    Stage4SelectionRouter,
    Stage4SelectionEngine,
    EconometricsFeaturePipeline,
)
from .calibration import (
    StandardPurgedCV,
    optimize_meta_threshold,
    compute_meta_confidence_sizing,
    compute_fractional_kelly_sizing,
    extract_primary_signals,
)
from .meta_engine import ProductionMetaTrainer
from .simulator import LowTurnoverMetaSimulator

__all__ = [
    "Layer1DataIntegrity",
    "Layer2MicrostructureTopologies",
    "Layer0TripleBarrier",
    "Stage2SingleAssetDGPScanner",
    "Layer10FeatureRouter",
    "Stage2DiagnosticEngine",
    "Stage3FeatureEngine",
    "Layer12RedundancyControl",
    "Layer13PredictiveDiagnostics",
    "Layer14LagTransformEngine",
    "Stage4SelectionRouter",
    "Stage4SelectionEngine",
    "EconometricsFeaturePipeline",
    "StandardPurgedCV",
    "optimize_meta_threshold",
    "compute_meta_confidence_sizing",
    "compute_fractional_kelly_sizing",
    "extract_primary_signals",
    "ProductionMetaTrainer",
    "LowTurnoverMetaSimulator",
]