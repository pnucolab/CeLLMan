from ._analysis import SSAMAnalysis, remove_outliers
from ._dataset import SSAMDataset

# Count-based spatial ingestion (Stereo-seq / Visium HD / StrataMap)
from ._griddataset import GridDataset
from ._panel import select_gene_panel, resolve_server, fetch_ontology_markers
from ._io import (aggregate_to_grid, load_stereoseq, load_visiumhd, load_strata,
                  load as load_count)

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
