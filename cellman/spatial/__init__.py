from ._analysis import SSAMAnalysis, remove_outliers
from ._dataset import SSAMDataset

__version__ = "0.1.0"

# SpatialData support utilities
try:
    import spatialdata
    _SPATIALDATA_AVAILABLE = True
except ImportError:
    _SPATIALDATA_AVAILABLE = False


def is_spatialdata_available():
    """Check if SpatialData is available."""
    return _SPATIALDATA_AVAILABLE
