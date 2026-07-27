import numcodecs
from multiprocessing.pool import ThreadPool
import pickle
import dask
import dask.array as da
import numpy as np
import pandas as pd
import zarr

import multiprocessing
import sys, os
import warnings
import gc

from sklearn import preprocessing
import scipy
from scipy import ndimage
from sklearn.decomposition import PCA
from tempfile import TemporaryDirectory
from sklearn.neighbors import kneighbors_graph
import igraph as ig
from sklearn.cluster import DBSCAN, OPTICS
from skimage import filters
from skimage.morphology import disk
from skimage import measure
import subprocess
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering

from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
import psutil
import math

# Import scanpy for clustering
try:
    import scanpy as sc
    import anndata as ad
    _SCANPY_AVAILABLE = True
except ImportError:
    _SCANPY_AVAILABLE = False
from sklearn.neighbors import LocalOutlierFactor
from tqdm import tqdm

import time

from scipy.ndimage import map_coordinates

from packaging import version

# Try to import utils module (C++ extension)
try:
    from .utils import check_avx512f, calc_corrmap, calc_kde, accumulate_gene, accumulate_gene_vector, normalize_vf
    _UTILS_AVAILABLE = True
except ImportError:
    _UTILS_AVAILABLE = False
    # Fallback implementations
    def check_avx512f():
        return False
    
    def calc_corrmap(*args, **kwargs):
        raise NotImplementedError("C++ extension not available. Please build SSAM properly.")
    
    def calc_kde(*args, **kwargs):
        raise NotImplementedError("C++ extension not available. Please build SSAM properly.")

# SpatialData support
try:
    import spatialdata as sd
    from spatialdata import SpatialData
    _SPATIALDATA_AVAILABLE = True
except ImportError:
    _SPATIALDATA_AVAILABLE = False
    SpatialData = None


def _binary_fill_holes_fast(mask):
    """``scipy.ndimage.binary_fill_holes`` with an optional CuPy backend.

    The GPU path (``cupyx.scipy.ndimage.binary_fill_holes``) uses the same
    default connectivity-1 structuring element as SciPy and is a purely
    boolean morphological operation, so the result is bit-identical to the
    CPU path. Any failure (CuPy missing, no CUDA device, out of memory)
    falls back transparently to SciPy. Set ``CELLMAN_DISABLE_GPU=1`` to
    force the CPU path.
    """
    if os.environ.get("CELLMAN_DISABLE_GPU"):
        return ndimage.binary_fill_holes(mask)
    try:
        import cupy as cp
        from cupyx.scipy.ndimage import binary_fill_holes as _gpu_fill
        d_mask = cp.asarray(mask)
        d_out = _gpu_fill(d_mask)
        out = cp.asnumpy(d_out)
        del d_mask, d_out
        cp.get_default_memory_pool().free_all_blocks()
        return out
    except Exception:
        # CuPy unavailable / no GPU / OOM -> identical CPU result.
        return ndimage.binary_fill_holes(mask)

if check_avx512f():
    from .utils import corr, calc_ctmap
else:
    corr = lambda a, b: np.corrcoef(a, b)[0, 1]
    def calc_ctmap(x, Y, _):
        mean_x = np.mean(x)
        mean_Y = np.mean(Y, axis=1)
        x_diff = x - mean_x
        Y_diff = Y - mean_Y[:, None]
        numerator = np.einsum('i,ji->j', x_diff, Y_diff)
        denominator = np.sqrt(np.einsum('i,i->', x_diff, x_diff) * np.einsum('ji,ji->j', Y_diff, Y_diff))
        correlation = np.where(denominator == 0, np.nan, numerator / denominator)
        return correlation


class MedoidCorrelation:
    def __init__(self, min_r=0.8):
        self.min_r = min_r

    def fit_predict(self, X):
        X = np.array(X, copy=True)
        labels = np.ones(X.shape[0], dtype=int)
        prev_midx = -1
        while True:
            vindices = np.where(labels > -1)[0]
            good_X = X[labels > -1]
            midx = vindices[np.argmin(np.sum(cdist(good_X, good_X, metric='correlation'), axis=0))]
            if midx == prev_midx:
                break
            prev_midx = midx
            m = X[midx]
            for vidx, v in zip(vindices, good_X):
                if corr(v, m) < self.min_r:
                    labels[vidx] = -1
        return labels


def remove_outliers(X, cluster_labels, outlier_detection_method='medoid-correlation', outlier_detection_kwargs={}, normalize=True):
    if outlier_detection_method == 'medoid-correlation':
        clf = MedoidCorrelation(**outlier_detection_kwargs)
    elif outlier_detection_method == 'robust-covariance':
        clf = EllipticEnvelope(**outlier_detection_kwargs)
    elif outlier_detection_method == 'one-class-svm':
        clf = OneClassSVM(**outlier_detection_kwargs)
    elif outlier_detection_method == 'isolation-forest':
        clf = IsolationForest(**outlier_detection_kwargs)
    elif outlier_detection_method == 'local-outlier-factor':
        clf = LocalOutlierFactor(**outlier_detection_kwargs)
    elif outlier_detection_method is None:
        return np.array(cluster_labels, copy=True)
    else:
        raise NotImplementedError("Method %s is not implemented."%outlier_detection_kwargs['method'])
    new_labels = np.array(cluster_labels, copy=True)
    if normalize:
        X = preprocessing.normalize(X)
    for cidx in np.unique(cluster_labels):
        if cidx == -1:
            continue
        cluster_indices = np.where(new_labels == cidx)[0]
        X_cl = X[cluster_indices]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predicted_labels = clf.fit_predict(X_cl)
        new_labels[cluster_indices[predicted_labels == -1]] = -1
    return new_labels
    
    
class SSAMAnalysis(object):
    """
    A class to run SSAM analysis.
    
    Supports both pandas.DataFrame and spatialdata.SpatialData as input data formats.
    
    For DataFrame input, required columns:
    - 'gene': gene names (can be index or column)
    - 'x', 'y': spatial coordinates in micrometers
    - 'z': depth coordinate (optional, for 3D data)
    
    For SpatialData input:
    - Points element with spatial coordinates and gene information
    - Automatically detects coordinate systems and gene annotation fields
    - Supports both 2D and 3D spatial data

    :param dataset: A SSAMDataset object.
    :type dataset: SSAMDataset
    :param ncores: Number of cores for parallel computation. If a negative value is given,
        ((# of all available cores on system) - abs(ncores)) cores will be used.
    :type ncores: int
    :param verbose: If True, then it prints out messages during the analysis.
    :type verbose: bool
    """
    def __init__(self, dataset, ncores=1, verbose=False):
        self.dataset = dataset
        if not ncores > 0:
            ncores += multiprocessing.cpu_count()
        if ncores > multiprocessing.cpu_count():
            ncores = multiprocessing.cpu_count()
        if not ncores > 0:
            raise ValueError("Invalid number of cores.")
        os.environ["OMP_NUM_THREADS"] = str(ncores)
        os.environ["OPENBLAS_NUM_THREADS"] = str(ncores)
        os.environ["MKL_NUM_THREADS"] = str(ncores)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(ncores)
        os.environ["NUMEXPR_NUM_THREADS"] = str(ncores)
        dask.config.set(pool=ThreadPool(ncores))
        self.ncores = ncores
        self.verbose = verbose

        

        # zarr 3's ``__contains__`` on Group can be unreliable for plain
        # arrays in some configurations, so collect keys explicitly once
        # and then check membership against that set throughout auto-load.
        try:
            zk = set(self.dataset.zarr_group.array_keys())
            zk |= set(self.dataset.zarr_group.group_keys())
        except Exception:
            try:
                zk = set(self.dataset.zarr_group.keys())
            except Exception:
                zk = set()

        if self.verbose:
            self._m(f"zarr keys: {sorted(zk) if zk else '(empty)'}")

        if 'kde_computed' in zk and all(self.dataset.zarr_group['kde_computed']):
            self._m("Loaded existing KDE results.")
            self._load_kde()

        # Load coordinate metadata if available
        if 'coordinate_offset' in zk:
            self.dataset.coordinate_offset = self.dataset.zarr_group['coordinate_offset'][:]
        if 'coordinate_bounds' in zk:
            self.dataset.coordinate_bounds = self.dataset.zarr_group['coordinate_bounds'][:]
        if 'original_coordinate_bounds' in zk:
            self.dataset.original_coordinate_bounds = self.dataset.zarr_group['original_coordinate_bounds'][:]

        if 'downsampled_vectors' in zk:
            self._m("Loaded existing local maxima.")
            self.dataset.downsampled_vectors = tuple(self.dataset.zarr_group['downsampled_vectors'][:])

        if 'normalized_vectors' in zk:
            self._m("Loaded a precomputed normalized vector.")
            self.dataset.normalized_vectors = self.dataset.zarr_group['normalized_vectors'][:]

        if 'genes' in zk:
            self._m("Loaded existing gene list.")
            self.dataset.genes = list(self.dataset.zarr_group['genes'][:])

        if 'vf_normalized' in zk:
            try :
                self._m("Loaded a precomputed normalized vector field.")
                self.dataset.vf_normalized = da.from_zarr(self.dataset.zarr_group['vf_normalized'])
            except:
                self._m("Failed to load precomputed normalized vector field. Perhaps the zarr data is in chunks.")

        if 'vf_scaled' in zk and 'scaled_vectors' in zk:
            self.dataset.vf_scaled = da.from_zarr(self.dataset.zarr_group['vf_scaled'])
            self.dataset.scaled_vectors = self.dataset.zarr_group['scaled_vectors'][:]

        # Each of cluster_labels / filtered_cluster_labels / centroids loaded
        # independently — store may have any subset (e.g. only cluster_labels).
        if 'cluster_labels' in zk:
            self._m("Loaded existing cluster labels.")
            self.dataset.cluster_labels = self.dataset.zarr_group['cluster_labels'][:]
        if 'filtered_cluster_labels' in zk:
            self.dataset.filtered_cluster_labels = self.dataset.zarr_group['filtered_cluster_labels'][:]
        if 'centroids' in zk:
            self.dataset.centroids = self.dataset.zarr_group['centroids'][:]
        if 'centroids_stdev' in zk:
            self.dataset.centroids_stdev = self.dataset.zarr_group['centroids_stdev'][:]
        
        if 'tsne' in zk:
            self._m("Loaded an existing t-SNE result.")
            self.dataset.tsne = self.dataset.zarr_group['tsne'][:]
        
        if 'umap' in zk:
            self._m("Loaded an existing UMAP result.")
            self.dataset.umap = self.dataset.zarr_group['umap'][:]
        
        if 'celltype_maps' in zk:
            self._m("Loaded existing cell type maps.")
            self.dataset.celltype_maps = self.dataset.zarr_group['celltype_maps'][:]
            if 'max_correlations' in zk:
                self.dataset.max_correlations = self.dataset.zarr_group['max_correlations'][:]
            else:
                self.dataset.max_correlations = None
            if 'max_probabilities' in zk:
                self.dataset.max_probabilities = self.dataset.zarr_group['max_probabilities'][:]
            else:
                self.dataset.max_probabilities = None

        if 'filtered_celltype_maps' in zk:
            self._m("Loaded existing filtered cell type maps.")
            self.dataset.filtered_celltype_maps = self.dataset.zarr_group['filtered_celltype_maps'][:]

        if 'celltype_binned_centers' in zk and 'celltype_binned_counts' in zk:
            self._m("Loaded existing cell type binned centers and counts.")
            self.dataset.celltype_binned_centers = self.dataset.zarr_group['celltype_binned_centers'][:]
            self.dataset.celltype_binned_counts = self.dataset.zarr_group['celltype_binned_counts'][:]
        
        if 'inferred_domains' in zk and 'inferred_domains_cells' in zk:
            self._m("Loaded existing inferred domains.")
            self.dataset.inferred_domains = self.dataset.zarr_group['inferred_domains'][:]
            self.dataset.inferred_domains_cells = self.dataset.zarr_group['inferred_domains_cells'][:]

        if 'inferred_domains_compositions' in self.dataset.zarr_group:
            self._m("Loaded existing inferred domains compositions.")
            self.dataset.inferred_domains_compositions = self.dataset.zarr_group['inferred_domains_compositions'][:]
        
        if 'watershed_segments' in self.dataset.zarr_group and 'watershed_celltype_maps' in self.dataset.zarr_group:
            self._m("Loaded existing watershed segmentations.")
            self.dataset.watershed_segments = self.dataset.zarr_group['watershed_segments'][:]
            self.dataset.watershed_celltype_maps = self.dataset.zarr_group['watershed_celltype_maps'][:]
        
        if 'transferred_cluster_labels' in self.dataset.zarr_group and 'transferred_cluster_correlations' in self.dataset.zarr_group:
            self._m("Loaded existing transferred cluster labels.")
            self.dataset.transferred_cluster_labels = self.dataset.zarr_group['transferred_cluster_labels'][:]
            self.dataset.transferred_cluster_correlations = self.dataset.zarr_group['transferred_cluster_correlations'][:]

    def _m(self, message):
        if self.verbose:
            print(message, flush=True)
    
    def _load_kde(self):
        assert 'kde_computed' in self.dataset.zarr_group, "KDE has not been computed!"
        assert all(self.dataset.zarr_group['kde_computed']), "KDE data is incomplete!"
        self.dataset.genes = list(self.dataset.zarr_group['genes'][:])
        if 'vf' in self.dataset.zarr_group:
            self.dataset._vf = da.from_zarr(self.dataset.zarr_group['vf'])
        self.dataset._vf_norm = da.from_zarr(self.dataset.zarr_group['vf_norm'])
        self.dataset.sampling_distance = self.dataset.zarr_group['vf_params'][0]
        self.dataset.bandwidth = self.dataset.zarr_group['vf_params'][1]
        self.dataset.shape = self.dataset.vf_norm.shape
        self.dataset.ndim = 2 if self.dataset.vf_norm.shape[-1] == 1 else 3
        self.dataset.expression_threshold = 1 / (np.sqrt(2 * np.pi) * self.dataset.bandwidth) ** self.dataset.ndim
        self.dataset.norm_threshold = self.dataset.expression_threshold * 2
        
    def _spatialdata_to_dataframe(self, sdata, element_name=None, coordinate_system=None):
        """
        Convert SpatialData object to DataFrame format compatible with SSAM.
        
        :param sdata: SpatialData object containing points data
        :type sdata: spatialdata.SpatialData
        :param element_name: Name of the points element to extract. If None, uses the first available.
        :type element_name: str
        :param coordinate_system: Coordinate system to use. If None, uses the first available.
        :type coordinate_system: str
        :returns: DataFrame with 'gene', 'x', 'y' columns (and 'z' if 3D) where gene is a column, not index
        :rtype: pandas.DataFrame
        """
        if not _SPATIALDATA_AVAILABLE:
            raise ImportError("SpatialData is not available. Install with: pip install spatialdata")
        
        if not isinstance(sdata, SpatialData):
            raise TypeError(f"Expected SpatialData object, got {type(sdata)}")
        
        # Get points data
        if element_name is None:
            if len(sdata.points) == 0:
                raise ValueError("No points data found in SpatialData object")
            element_name = list(sdata.points.keys())[0]
        
        if element_name not in sdata.points:
            raise ValueError(f"Points element '{element_name}' not found. Available: {list(sdata.points.keys())}")
        
        points_df = sdata.points[element_name]
        
        # Convert Dask DataFrame to pandas DataFrame if necessary
        if hasattr(points_df, 'compute'):
            points_df = points_df.compute()
        
        # Determine coordinate system
        if coordinate_system is None:
            if hasattr(points_df, 'attrs') and 'transform' in points_df.attrs:
                coord_systems = list(points_df.attrs['transform'].keys()) if points_df.attrs['transform'] else []
                if coord_systems:
                    coordinate_system = coord_systems[0]
        
        # Extract spatial coordinates
        spatial_coords = ['x', 'y']
        if 'z' in points_df.columns:
            spatial_coords.append('z')
        
        # Check required columns
        if not all(coord in points_df.columns for coord in spatial_coords[:2]):
            raise ValueError(f"Required spatial coordinates 'x', 'y' not found in points data. Available columns: {points_df.columns.tolist()}")
        
        # Look for gene information
        gene_col = None
        possible_gene_cols = ['gene', 'gene_symbol', 'feature_name', 'target']
        for col in possible_gene_cols:
            if col in points_df.columns:
                gene_col = col
                break
        
        if gene_col is None:
            raise ValueError(f"No gene information found. Expected one of: {possible_gene_cols}. Available columns: {points_df.columns.tolist()}")
        
        # Filter to real genes only (exclude DeprecatedCodeword, NegControl, etc.)
        if 'is_gene' in points_df.columns:
            points_df = points_df[points_df['is_gene'] == True]

        # Create output DataFrame
        result_cols = [gene_col] + spatial_coords
        result_df = points_df[result_cols].copy()
        
        # Handle Categorical columns - convert to string
        if pd.api.types.is_categorical_dtype(result_df[gene_col]):
            result_df[gene_col] = result_df[gene_col].astype(str)
        
        # Ensure coordinate columns are numeric
        for coord in spatial_coords:
            if not pd.api.types.is_numeric_dtype(result_df[coord]):
                result_df[coord] = pd.to_numeric(result_df[coord], errors='coerce')
        
        # Rename gene column to 'gene' if necessary
        if gene_col != 'gene':
            result_df = result_df.rename(columns={gene_col: 'gene'})
        
        return result_df

    def _validate_input_data(self, data, element_name=None, width=None, height=None, depth=None):
        """
        Validate and normalize input data for SSAM analysis.
        
        :param data: Input data - either DataFrame or SpatialData object
        :param element_name: Name of the points element to extract from SpatialData
        :param width: Image width in um (optional, auto-detected if not provided)
        :param height: Image height in um (optional, auto-detected if not provided)  
        :param depth: Image depth in um (optional, auto-detected if not provided)
        :returns: Tuple of (DataFrame, width, height, depth, is_3d, coordinate_metadata)
        """
        # Handle SpatialData input
        if _SPATIALDATA_AVAILABLE and isinstance(data, SpatialData):
            # Convert SpatialData to DataFrame
            df = self._spatialdata_to_dataframe(data, element_name)
            
            # Calculate actual coordinate ranges
            x_min, x_max = df['x'].min(), df['x'].max()
            y_min, y_max = df['y'].min(), df['y'].max()
            z_min = z_max = 0
            if 'z' in df.columns:
                z_min, z_max = df['z'].min(), df['z'].max()
            
            # Calculate actual dimensions
            actual_width = float(x_max - x_min)
            actual_height = float(y_max - y_min)
            actual_depth = float(z_max - z_min) if 'z' in df.columns else 1
            
            # Check if 3D data (z dimension meaningful)
            z_threshold = 1.0  # 1μm threshold
            is_3d = ('z' in df.columns and 
                    actual_depth > z_threshold and 
                    df['z'].nunique() > 1)
            
            # If user didn't specify dimensions, use auto-detected values
            if width is None:
                width = actual_width
            if height is None:
                height = actual_height
            if depth is None and is_3d:
                depth = actual_depth
            elif depth is None:
                depth = 1
                
            # Normalize coordinates to start from (0,0,0)
            df_normalized = df.copy()
            df_normalized['x'] = df_normalized['x'] - x_min
            df_normalized['y'] = df_normalized['y'] - y_min
            if 'z' in df_normalized.columns:
                df_normalized['z'] = df_normalized['z'] - z_min
            
            # Create coordinate metadata
            coordinate_metadata = {
                'offset': [float(x_min), float(y_min), float(z_min)],
                'bounds': [width, height, depth],
                'original_bounds': [[float(x_min), float(x_max)], 
                                [float(y_min), float(y_max)], 
                                [float(z_min), float(z_max)]]
            }
                
            return df_normalized, width, height, depth, is_3d, coordinate_metadata
        
        # Handle DataFrame input (existing functionality)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
            
            # Ensure proper index format
            if df.index.name != 'gene':
                if 'gene' in df.columns:
                    df = df.set_index('gene')
                else:
                    raise ValueError("DataFrame must have 'gene' column or 'gene' as index")
            
            # Validate required columns
            required_cols = ['x', 'y']
            if depth is not None and depth > 1:
                required_cols.append('z')
                
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Calculate actual coordinate ranges
            x_min, x_max = df['x'].min(), df['x'].max()
            y_min, y_max = df['y'].min(), df['y'].max()
            z_min = z_max = 0
            if 'z' in df.columns:
                z_min, z_max = df['z'].min(), df['z'].max()
            
            # Calculate actual dimensions
            actual_width = float(x_max - x_min)
            actual_height = float(y_max - y_min)
            actual_depth = float(z_max - z_min) if 'z' in df.columns else 1
            
            # Check if 3D data
            z_threshold = 1.0  # 1μm threshold
            is_3d = ('z' in df.columns and 
                    actual_depth > z_threshold and 
                    df['z'].nunique() > 1)
            
            # Auto-detect dimensions if not provided
            if width is None:
                width = actual_width
            if height is None:
                height = actual_height
            if depth is None and is_3d:
                depth = actual_depth
            elif depth is None:
                depth = 1
                
            # Normalize coordinates to start from (0,0,0)
            df_normalized = df.copy()
            df_normalized['x'] = df_normalized['x'] - x_min
            df_normalized['y'] = df_normalized['y'] - y_min
            if 'z' in df_normalized.columns:
                df_normalized['z'] = df_normalized['z'] - z_min
            
            # Create coordinate metadata
            coordinate_metadata = {
                'offset': [float(x_min), float(y_min), float(z_min)],
                'bounds': [width, height, depth],
                'original_bounds': [[float(x_min), float(x_max)], 
                                [float(y_min), float(y_max)], 
                                [float(z_min), float(z_max)]]
            }
                
            return df_normalized, width, height, depth, is_3d, coordinate_metadata
        
        else:
            raise TypeError(f"Input data must be pandas.DataFrame or spatialdata.SpatialData, got {type(data)}")
        
    def calc_chunk(self, vx, vy, vz, g, mode="kde", rho=1.0, avail_frac=0.1, g2=None):
        avail = psutil.virtual_memory().available
        if mode == "map_celltypes_cpu":
            if g2 is None:
                g2 = g
            bytes_per_voxel = 4*g + 12 + rho*(4*g + 12*g2 + 40)
        else:
            bytes_per_voxel = 4*g

        bytes_per_xslice = vy * vz * bytes_per_voxel
        cx = max(1, min(vx, int(avail * avail_frac / bytes_per_xslice)))
        n_chunk = math.ceil(vx / cx)
        return cx, n_chunk


    def _pick_clustering_backend(self, auto_gpu=True):
        """Return (backend_module, name) for neighbor graph + Leiden/Louvain.

        - If ``auto_gpu`` and ``rapids_singlecell`` imports successfully and
          a CUDA device is available, use rapids_singlecell (GPU).
        - Otherwise, fall back to scanpy (CPU).

        Both modules expose the same ``.pp.neighbors`` and
        ``.tl.leiden`` / ``.tl.louvain`` APIs, so the caller can use the
        returned module interchangeably.
        """
        if auto_gpu:
            try:
                import rapids_singlecell as _rsc
                cuda_ok = False
                try:
                    import torch
                    cuda_ok = bool(torch.cuda.is_available())
                except ImportError:
                    try:
                        import cupy
                        cuda_ok = cupy.cuda.runtime.getDeviceCount() > 0
                    except Exception:
                        cuda_ok = False
                if cuda_ok:
                    return _rsc, 'rapids_singlecell (GPU)'
            except ImportError:
                pass
        if not _SCANPY_AVAILABLE:
            raise ImportError(
                "Neither rapids_singlecell (GPU) nor scanpy (CPU) is available. "
                "Install one of them for clustering."
            )
        return sc, 'scanpy (CPU)'

    def _resolve_cluster_vectors(self, vectors=None):
        """Feature matrix for outlier / centroid computation: an explicit
        ``vectors`` (e.g. a clustering AnnData's ``X``) wins, else the dataset's
        scaled/normalized vectors."""
        if vectors is not None:
            return np.asarray(vectors)
        if getattr(self.dataset, 'scaled_vectors', None) is not None:
            return self.dataset.scaled_vectors
        if getattr(self.dataset, 'normalized_vectors', None) is not None:
            return self.dataset.normalized_vectors
        raise ValueError("Neither scaled_vectors nor normalized_vectors are available. "
                         "Please run downsample_vector() first.")

    def _correct_cluster_labels(self, cluster_labels, outlier_detection_method,
                                outlier_detection_kwargs, vectors=None):
        vecs = self._resolve_cluster_vectors(vectors)
        return remove_outliers(vecs, cluster_labels, outlier_detection_method,
                               outlier_detection_kwargs)

    def _calc_centroid(self, cluster_labels, normalize=True, norm='l2', vectors=None):
        vecs = self._resolve_cluster_vectors(vectors)
        centroids = []
        centroids_stdev = []
        #medoids = []
        for lbl in np.unique(cluster_labels):
            if lbl == -1:
                continue
            cl_vecs = vecs[cluster_labels == lbl, :]
            if normalize:
                preprocessing.normalize(cl_vecs, norm=norm, axis=1)
            #cl_dists = scipy.spatial.distance.cdist(cl_vecs, cl_vecs, metric)
            #medoid = cl_vecs[np.argmin(np.sum(cl_dists, axis=0))]
            centroid = np.mean(cl_vecs, axis=0)
            centroid_stdev = np.std(cl_vecs, axis=0)
            #medoids.append(medoid)
            centroids.append(centroid)
            centroids_stdev.append(centroid_stdev)
        return centroids, centroids_stdev#, medoids

    def cluster_vectors(self, adata, *, method="leiden", resolution=1.0,
                        key_added="leiden", min_cluster_size=2, max_correlation=1.0,
                        outlier_detection_method='medoid-correlation',
                        outlier_detection_kwargs={}, random_state=0, auto_gpu=True,
                        neighbors_key=None, **kwargs):
        """Cluster a downsampled-vector AnnData IN PLACE (scanpy-style).

        Mirrors the ``sc.pp.pca`` -> ``sc.pp.neighbors`` -> cluster idiom: PCA and the
        kNN graph are the CALLER's job (run scanpy or rapids-singlecell on ``adata``
        first), and this method does ONLY clustering + centroid computation::

            adata = analysis.downsample_vector(...)      # AnnData of normalized_vectors
            sc.pp.pca(adata); sc.pp.neighbors(adata)     # or the rapids equivalents
            analysis.cluster_vectors(adata)              # writes adata.obs['leiden']

        Runs Leiden/Louvain on ``adata``'s existing neighbor graph, refines the labels
        (outlier removal, small-cluster drop, optional correlation merge), computes
        per-cluster CENTROIDS from ``adata.X`` (= normalized_vectors, the same space as
        ``vf_normalized`` for :meth:`map_celltypes`), then

          * writes the final labels to ``adata.obs[key_added]`` (categorical), and
          * persists ``cluster_labels`` / ``filtered_cluster_labels`` / ``centroids`` /
            ``centroids_stdev`` to the zarr store so ``map_celltypes`` can read them.

        Returns ``None`` (mutates ``adata``), like ``sc.pp.*`` / ``sc.tl.leiden``.

        Parameters
        ----------
        adata : anndata.AnnData
            Downsampled vectors (from :meth:`downsample_vector`) with a neighbor graph
            already computed. ``adata.X`` must be the normalized vectors, row-aligned
            with the store's ``normalized_vectors`` (do not reorder/subset if you will
            call ``map_celltypes(method='tsvm')`` afterwards).
        method : {'leiden', 'louvain'}, default 'leiden'
        resolution : float, default 1.0
        key_added : str, default 'leiden'
            ``adata.obs`` column the final labels are written to.
        neighbors_key : str, optional
            Select a non-default neighbor graph (forwarded to the backend).
        min_cluster_size, max_correlation, outlier_detection_method,
        outlier_detection_kwargs :
            Label-refinement controls (unchanged from the previous implementation).
        """
        import scipy.sparse as _sp
        if adata is None or not hasattr(adata, "obs"):
            raise TypeError(
                "cluster_vectors now takes the downsampled AnnData as its first "
                "argument: analysis.cluster_vectors(adata). Build it with "
                "adata = analysis.downsample_vector(...), then run sc.pp.pca(adata) "
                "and sc.pp.neighbors(adata) before clustering.")
        has_graph = (neighbors_key is not None) or ('neighbors' in adata.uns) \
            or ('connectivities' in adata.obsp)
        if not has_graph:
            raise ValueError(
                "cluster_vectors expects a kNN graph on `adata`; run sc.pp.pca(adata) "
                "+ sc.pp.neighbors(adata) (or the rapids-singlecell equivalents) first, "
                "as you would before sc.tl.leiden.")

        def remove_small_clusters(lbls, lbls2=None):
            small_clusters = []
            cluster_indices = []
            lbls = np.array(lbls)
            for lbl in np.unique(lbls):
                if lbl == -1:
                    continue
                cnt = np.sum(lbls == lbl)
                if cnt < min_cluster_size:
                    small_clusters.append(lbl)
                else:
                    cluster_indices.append(lbl)
            for lbl in small_clusters:
                lbls[lbls == lbl] = -1
            tmp = np.array(lbls, copy=True)
            for i, idx in enumerate(cluster_indices):
                lbls[tmp == idx] = i
            if lbls2 is not None:
                for lbl in small_clusters:
                    lbls2[lbls2 == lbl] = -1
                tmp = np.array(lbls2, copy=True)
                for i, idx in enumerate(cluster_indices):
                    lbls2[tmp == idx] = i
                return lbls, lbls2
            return lbls

        # normalized vectors = adata.X (same space as vf_normalized -> map_celltypes)
        X = adata.X
        X = X.toarray() if _sp.issparse(X) else np.asarray(X)
        X = X.astype(np.float32)

        # ---- Leiden/Louvain on the caller-built neighbor graph ----
        backend, backend_name = self._pick_clustering_backend(auto_gpu=auto_gpu)
        self._m(f"cluster_vectors: backend = {backend_name}")
        nbrs_kw = {'neighbors_key': neighbors_key} if neighbors_key is not None else {}
        tmp_key = "__cellman_cluster_raw"

        def _run(mod):
            if method == 'leiden':
                mod.tl.leiden(adata, resolution=resolution, random_state=random_state,
                              key_added=tmp_key, **nbrs_kw)
            elif method == 'louvain':
                mod.tl.louvain(adata, resolution=resolution, random_state=random_state,
                               key_added=tmp_key, **nbrs_kw)
            else:
                raise NotImplementedError(
                    f"Method '{method}' is not supported. Only 'leiden' and 'louvain'.")
        try:
            _run(backend)
        except Exception as e:  # noqa: BLE001
            if backend is not sc and _SCANPY_AVAILABLE:
                self._m(f"cluster_vectors: {backend_name} clustering failed ({e}); "
                        f"falling back to scanpy (CPU)")
                _run(sc)
            else:
                raise
        all_lbls = np.asarray(adata.obs[tmp_key].astype(int).values)
        del adata.obs[tmp_key]

        # ---- label refinement + centroids (from adata.X) ----
        if outlier_detection_method is not None:
            filtered_all_lbls = self._correct_cluster_labels(
                all_lbls, outlier_detection_method, outlier_detection_kwargs, vectors=X)
            filtered_all_lbls, all_lbls = remove_small_clusters(filtered_all_lbls, all_lbls)
        else:
            filtered_all_lbls = all_lbls = remove_small_clusters(all_lbls)

        centroids, centroids_stdev = self._calc_centroid(filtered_all_lbls, vectors=X)

        if max_correlation < 1.0:
            Z = scipy.cluster.hierarchy.linkage(centroids, metric='correlation')
            clbls = scipy.cluster.hierarchy.fcluster(Z, 1 - max_correlation, 'distance')
            merge_candidates = []
            for i in set(clbls):
                leaf_indices = np.where(clbls == i)[0]
                if len(leaf_indices) > 1:
                    merge_candidates.append(leaf_indices)
            removed_indices = []
            for cand in merge_candidates:
                for i in cand[1:]:
                    all_lbls[all_lbls == i] = cand[0]
                    removed_indices.append(i)
            for i in sorted(removed_indices, reverse=True):
                all_lbls[all_lbls > i] -= 1
            if outlier_detection_method is not None:
                filtered_all_lbls = self._correct_cluster_labels(
                    all_lbls, outlier_detection_method, outlier_detection_kwargs, vectors=X)
                filtered_all_lbls, all_lbls = remove_small_clusters(filtered_all_lbls, all_lbls)
            else:
                filtered_all_lbls = all_lbls = remove_small_clusters(all_lbls)
            centroids, centroids_stdev = self._calc_centroid(filtered_all_lbls, vectors=X)

        # ---- persist labels + centroids to the store (map_celltypes reads these) ----
        self.dataset.cluster_labels = all_lbls
        self.dataset.filtered_cluster_labels = filtered_all_lbls
        self.dataset.centroids = np.array(centroids)
        self.dataset.centroids_stdev = np.array(centroids_stdev)
        for _k in ('cluster_labels', 'filtered_cluster_labels', 'centroids', 'centroids_stdev'):
            if _k in self.dataset.zarr_group:
                del self.dataset.zarr_group[_k]
        self.dataset.zarr_group['cluster_labels'] = self.dataset.cluster_labels
        self.dataset.zarr_group['filtered_cluster_labels'] = self.dataset.filtered_cluster_labels
        self.dataset.zarr_group['centroids'] = self.dataset.centroids
        self.dataset.zarr_group['centroids_stdev'] = self.dataset.centroids_stdev

        # ---- write final labels into adata IN PLACE (scanpy-style) ----
        adata.obs[key_added] = pd.Categorical(np.asarray(filtered_all_lbls).astype(str))
        self._m("Found %d clusters" % len(centroids))
        return None

    def _downsampled_adata(self):
        """Build the AnnData that :meth:`downsample_vector` returns: ``X`` =
        normalized_vectors, ``obsm['spatial']`` = downsampled positions,
        ``obs['vf_norm']`` = per-voxel local density, ``var`` = genes. Feed it to
        ``sc.pp.pca`` / ``sc.pp.neighbors`` / :meth:`cluster_vectors`."""
        ds = self.dataset
        nv = np.asarray(ds.normalized_vectors, dtype=np.float32)
        genes = getattr(ds, 'genes', None)
        if genes is None and getattr(ds, 'zarr_store', None) is not None:
            store_path = ds._zarr_store_path(ds.zarr_store)
            genes = zarr.open(os.path.join(store_path, 'genes'), mode='r')[:]
            ds.genes = genes
        if genes is None:
            genes = [f'g{i}' for i in range(nv.shape[1])]
        genes = [g.decode() if isinstance(g, bytes) else str(g) for g in genes]
        adata = ad.AnnData(
            X=nv,
            obs=pd.DataFrame(index=[f'v5_{i}' for i in range(nv.shape[0])]),
            var=pd.DataFrame(index=pd.Index(genes, name='gene')),
        )
        dv = getattr(ds, 'downsampled_vectors', None)
        if dv is not None:
            adata.obsm['spatial'] = np.stack(
                [np.asarray(dv[0]), np.asarray(dv[1])], axis=1).astype(np.float32)
        vfn = getattr(ds, 'downsampled_vf_norm', None)
        if vfn is not None and len(vfn) == adata.n_obs:
            adata.obs['vf_norm'] = np.asarray(vfn, dtype=np.float32)
        return adata

    def _prepare_grid(self, sdata, element_name=None, bandwidth=2.5, sampling_distance=1.0,
                      prune_coefficient=4.3, size_after_normalization=10):
        """
        Parse SpatialData, compute grid dimensions (vx, vy, vz), and store them in
        ``self.dataset.shape``.  Also persists coordinate metadata to zarr.

        Returns
        -------
        vx, vy, vz : int
        """
        locations, width, height, depth, is_3d, coord_metadata = self._validate_input_data(
            sdata, element_name, None, None, None
        )

        self.dataset.coordinate_offset = coord_metadata['offset']
        self.dataset.coordinate_bounds = coord_metadata['bounds']
        self.dataset.original_coordinate_bounds = coord_metadata['original_bounds']
        self.dataset.sampling_distance = sampling_distance
        self.bandwidth = bandwidth
        self.prune_coefficient = prune_coefficient
        self.size_after_normalization = size_after_normalization

        if locations.index.name != 'gene':
            locations = locations.set_index('gene')
        if is_3d and depth > 1:
            self.locations = locations.reindex(['x', 'y', 'z'], axis=1)
        else:
            self.locations = locations.reindex(['x', 'y'], axis=1)

        grid_xyz = np.ceil(np.array([width, height, depth]) / sampling_distance).astype(int)
        vx, vy, vz = int(grid_xyz[0]), int(grid_xyz[1]), int(grid_xyz[2])
        self.dataset.shape = (vx, vy, vz)

        for name, value in [
            ('coordinate_offset', np.array(self.dataset.coordinate_offset)),
            ('coordinate_bounds', np.array(self.dataset.coordinate_bounds)),
            ('original_coordinate_bounds', np.array(self.dataset.original_coordinate_bounds)),
        ]:
            if name in self.dataset.zarr_group:
                del self.dataset.zarr_group[name]
            self.dataset.zarr_group[name] = value

        return vx, vy, vz

    def _parse_transcripts(self, sdata, sampling_distance=1.0, filter_is_gene=True):
        """
        Extract transcript coordinates from SpatialData and compute grid indices.

        Returns
        -------
        meta : dict
            Keys: gene_names, coordinate_offset, tx, ty, tz,
                  xc_g, yc_g, zc_g, is_3d, dim
        """
        pts = sdata.points[list(sdata.points.keys())[0]]
        if hasattr(pts, 'compute'):
            pts = pts.compute()
        if filter_is_gene and 'is_gene' in pts.columns:
            pts = pts[pts['is_gene'] == True]

        gene_col = next(
            (c for c in ['gene', 'gene_symbol', 'feature_name', 'target'] if c in pts.columns),
            None,
        )
        if gene_col is None:
            raise ValueError(f'gene column not found. columns: {pts.columns.tolist()}')

        gene_names = pts[gene_col].astype(str).values
        tx_raw = pts['x'].values.astype(np.float64)
        ty_raw = pts['y'].values.astype(np.float64)
        has_z = 'z' in pts.columns
        tz_raw = pts['z'].values.astype(np.float64) if has_z else np.zeros(len(tx_raw), dtype=np.float64)

        x_min, y_min, z_min = tx_raw.min(), ty_raw.min(), tz_raw.min()
        tx = tx_raw - x_min
        ty = ty_raw - y_min
        tz = tz_raw - z_min

        z_threshold = 1.0
        is_3d = has_z and (tz_raw.max() - tz_raw.min()) > z_threshold and len(set(tz_raw[:10000])) > 1
        dim = 3 if is_3d else 2

        xc_g = np.floor(tx / sampling_distance).astype(np.int64)
        yc_g = np.floor(ty / sampling_distance).astype(np.int64)
        zc_g = np.floor(tz / sampling_distance).astype(np.int64) if is_3d else np.zeros(len(tx), dtype=np.int64)

        return {
            'gene_names': gene_names,
            'coordinate_offset': [float(x_min), float(y_min), float(z_min)],
            'tx': tx, 'ty': ty, 'tz': tz,
            'xc_g': xc_g, 'yc_g': yc_g, 'zc_g': zc_g,
            'is_3d': is_3d, 'dim': dim,
        }

    # def _compute_normalized_vectors(self, query_voxels, meta, genes=None,
    #                                  bandwidth=2.5, sampling_distance=1.0,
    #                                  prune_coefficient=4.3, size_after_normalization=10,
    #                                  log_transform=True):
    #     """
    #     Compute L1-normalised (+ optional log1p) KDE vectors at arbitrary voxel positions.
    #
    #     Parameters
    #     ----------
    #     query_voxels : ndarray, shape (N, 2|3)  — grid-index coordinates
    #     meta : dict returned by ``_parse_transcripts``
    #
    #     Returns
    #     -------
    #     normalized : ndarray (N, n_genes)
    #     vf_norm    : ndarray (N,)
    #     genes      : list[str]
    #     coordinate_offset : list[float]
    #     """
    #     from scipy.spatial import cKDTree
    #
    #     gene_names = meta['gene_names']
    #     tx, ty, tz = meta['tx'], meta['ty'], meta['tz']
    #     is_3d = meta['is_3d']
    #     dim = meta['dim']
    #     coordinate_offset = meta['coordinate_offset']
    #
    #     if genes is None:
    #         genes = sorted(np.unique(gene_names))
    #     gene_to_idx = {g: i for i, g in enumerate(genes)}
    #     n_genes = len(genes)
    #
    #     gidx_arr = np.array([gene_to_idx.get(g, -1) for g in gene_names], dtype=np.int32)
    #     valid = gidx_arr >= 0
    #     tx, ty, tz = tx[valid], ty[valid], tz[valid]
    #     gidx_arr = gidx_arr[valid]
    #
    #     query_voxels = np.asarray(query_voxels, dtype=np.float64)
    #     qx = query_voxels[:, 0] * sampling_distance
    #     qy = query_voxels[:, 1] * sampling_distance
    #     qz = (query_voxels[:, 2] * sampling_distance
    #           if (is_3d and query_voxels.shape[1] > 2)
    #           else np.zeros(len(qx)))
    #
    #     norm_factor = (2.0 * np.pi * bandwidth ** 2) ** (dim / 2.0)
    #     two_h2 = 2.0 * bandwidth ** 2
    #     rw = bandwidth * prune_coefficient
    #     rw2 = rw * rw
    #     rp = int(np.floor(rw / sampling_distance + 1e-12))
    #
    #     if is_3d:
    #         grid_coords = np.column_stack([meta['xc_g'][valid], meta['yc_g'][valid], meta['zc_g'][valid]])
    #         query_grid = query_voxels[:, :3]
    #         transcript_coords = np.column_stack([tx, ty, tz])
    #         query_coords = np.column_stack([qx, qy, qz])
    #     else:
    #         grid_coords = np.column_stack([meta['xc_g'][valid], meta['yc_g'][valid]])
    #         query_grid = query_voxels[:, :2]
    #         transcript_coords = np.column_stack([tx, ty])
    #         query_coords = np.column_stack([qx, qy])
    #
    #     grid_tree = cKDTree(grid_coords.astype(np.float64))
    #     neighbors_list = grid_tree.query_ball_point(query_grid.astype(np.float64), r=rp, p=np.inf)
    #
    #     n_query = len(query_voxels)
    #     kde_values = np.zeros((n_query, n_genes), dtype=np.float32)
    #
    #     for i, neighbors in enumerate(neighbors_list):
    #         if not neighbors:
    #             continue
    #         nb = np.asarray(neighbors, dtype=np.int64)
    #         diff = transcript_coords[nb] - query_coords[i]
    #         r2 = (diff ** 2).sum(axis=1)
    #         mask = r2 <= rw2
    #         if not mask.any():
    #             continue
    #         w = (np.exp(-r2[mask] / two_h2) / norm_factor).astype(np.float32)
    #         np.add.at(kde_values[i], gidx_arr[nb[mask]], w)
    #
    #     vf_norm = kde_values.sum(axis=1)
    #     normalized = np.zeros_like(kde_values)
    #     nonzero = vf_norm > 0
    #     normalized[nonzero] = (kde_values[nonzero] / vf_norm[nonzero, np.newaxis]
    #                            * size_after_normalization)
    #     if log_transform:
    #         np.log1p(normalized, out=normalized)
    #
    #     return normalized, vf_norm, genes, coordinate_offset

    def _compute_normalized_vectors(self, query_voxels, meta, genes=None,
                                     bandwidth=2.5, sampling_distance=1.0,
                                     prune_coefficient=4.3, size_after_normalization=10,
                                     log_transform=True):
        """
        Compute L1-normalised (+ optional log1p) KDE vectors at arbitrary voxel positions.

        Uses the C++ ``kde_at_points`` function (grid-cell hash map + OpenMP)
        which produces results identical to ``accumulate_gene_vector`` at the
        same coordinates.

        Parameters
        ----------
        query_voxels : ndarray, shape (N, 2|3)  — grid-index coordinates
        meta : dict returned by ``_parse_transcripts``

        Returns
        -------
        normalized : ndarray (N, n_genes)
        vf_norm    : ndarray (N,)
        genes      : list[str]
        coordinate_offset : list[float]
        """
        from .utils import kde_at_points

        gene_names = meta['gene_names']
        tx, ty, tz = meta['tx'], meta['ty'], meta['tz']
        is_3d = meta['is_3d']
        dim = meta['dim']
        coordinate_offset = meta['coordinate_offset']

        if genes is None:
            genes = sorted(np.unique(gene_names))
        gene_to_idx = {g: i for i, g in enumerate(genes)}
        n_genes = len(genes)

        gidx_arr = np.array([gene_to_idx.get(g, -1) for g in gene_names], dtype=np.int32)
        valid = gidx_arr >= 0
        tx, ty, tz = tx[valid], ty[valid], tz[valid]
        gidx_arr = gidx_arr[valid]

        query_voxels = np.asarray(query_voxels, dtype=np.float64)
        qx = (query_voxels[:, 0] * sampling_distance).astype(np.float64)
        qy = (query_voxels[:, 1] * sampling_distance).astype(np.float64)
        qz = ((query_voxels[:, 2] * sampling_distance).astype(np.float64)
              if (is_3d and query_voxels.shape[1] > 2)
              else np.zeros(len(qx), dtype=np.float64))

        kde_values = kde_at_points(
            qx, qy, qz,
            tx.astype(np.float64), ty.astype(np.float64), tz.astype(np.float64),
            gidx_arr, n_genes,
            float(sampling_distance), float(bandwidth), float(prune_coefficient),
            int(dim), int(self.ncores),
        )

        vf_norm = kde_values.sum(axis=1)
        normalized = np.zeros_like(kde_values)
        nonzero = vf_norm > 0
        normalized[nonzero] = (kde_values[nonzero] / vf_norm[nonzero, np.newaxis]
                               * size_after_normalization)
        if log_transform:
            np.log1p(normalized, out=normalized)

        return normalized, vf_norm, genes, coordinate_offset

    # -------------------------------------------------------------------------
    # H&E downsampling helpers
    # -------------------------------------------------------------------------

    def _he_array_from_sdata(self, sdata, image_key='he'):
        """Extract an (H, W, 3) uint8 RGB ndarray from ``sdata.images[image_key]``.

        Handles both single-scale (``DataArray``) and multiscale
        (``DataTree``) representations as written by spatialdata-io / NGFF.
        For multiscale, the highest-resolution level is returned.

        RGB(A) images stored channel-first ``(c, y, x)`` are transposed to
        ``(y, x, c)``. Single-channel images — e.g. a fluorescence morphology
        MIP stored as ``(1, y, x)`` or ``(y, x, 1)`` — are squeezed to 2-D and
        replicated across 3 channels. Higher-bit-depth integer images (e.g.
        uint16 fluorescence) are contrast-stretched per image to 0-255 (99.9th
        percentile → 255) rather than clipped, so tissue spans the dynamic
        range instead of saturating; uint8 RGB (H&E) is returned unchanged.

        The caller is responsible for ensuring that the H&E element in
        ``sdata`` already spatially corresponds to the morphology image
        (i.e., cropped to the sequenced region and rasterized onto an
        axis-aligned grid). This function does not apply any
        coordinate-system transform.
        """
        if image_key not in sdata.images:
            raise KeyError(
                f"sdata.images does not contain key '{image_key}'. "
                f"Available: {list(sdata.images.keys())}"
            )
        node = sdata.images[image_key]

        # Multiscale (DataTree): pick the full-resolution leaf
        if not hasattr(node, 'values') or hasattr(node, 'children'):
            scale_keys = list(node.keys())
            if not scale_keys:
                raise RuntimeError(
                    f"H&E multiscale image at images['{image_key}'] has no scales."
                )
            leaf = node[scale_keys[0]]
            if hasattr(leaf, 'image'):
                arr = leaf['image'].values
            else:
                data_vars = list(leaf.data_vars)
                if not data_vars:
                    raise RuntimeError(
                        f"H&E multiscale leaf '{scale_keys[0]}' has no data_vars."
                    )
                arr = leaf[data_vars[0]].values
        else:
            arr = node.values

        # ---- Collapse the channel axis to (y, x) or (y, x, 3/4) ----
        # spatialdata stores images channel-first as (c, y, x). RGB(A) -> move
        # channels last; single-channel (1, y, x) or (y, x, 1) -> squeeze to 2-D
        # (handled below by the grayscale replication). Without the single-
        # channel case a (1, y, x) morphology MIP would slip past moveaxis and
        # get sliced along x by ``arr[:, :, :3]`` into a (1, y, 3) garbage array.
        # Multi-channel fluorescence morphology (e.g. MERSCOPE Cellbound stores
        # 2 stain channels as (2, y, x)) matches none of the RGB(A)/single
        # branches; without an explicit case the channel axis is never moved and
        # the final ``arr[:, :, :3]`` slices along x into a (c, y, 3) strip
        # (observed: (2, 20000, 3)), which the bbox crop then pads to an almost
        # entirely blank tile. Collapse such stacks to a tissue-bright (y, x)
        # intensity via a max projection over the channel axis.
        if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
            arr = np.moveaxis(arr, 0, -1)        # (3|4, y, x) -> (y, x, 3|4)
        elif arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]                          # (1, y, x) -> (y, x)
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]                     # (y, x, 1) -> (y, x)
        elif arr.ndim == 3 and arr.shape[0] < arr.shape[-1] and arr.shape[0] not in (3, 4):
            # channel-first multi-channel (c, y, x), c not 1/3/4 -> max-project
            arr = arr.max(axis=0)                 # (c, y, x) -> (y, x)
        elif arr.ndim == 3 and arr.shape[-1] not in (3, 4) and arr.shape[-1] < arr.shape[0]:
            # channel-last multi-channel (y, x, c), c not 1/3/4 -> max-project
            arr = arr.max(axis=-1)                # (y, x, c) -> (y, x)

        # ---- Cast to uint8 on the 2-D/RGB array (before grayscale stacking) ----
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.0:
                # float image already normalised to [0, 1]
                arr = (arr.astype(np.float32) * 255.0)
            else:
                # Higher-bit-depth integer (e.g. uint16 fluorescence) or float
                # beyond [0, 1]: contrast-stretch per image to 0-255 using the
                # 99.9th percentile (robust to hot pixels) instead of clipping
                # to 255, which would saturate tissue and collapse the Otsu
                # histogram. Percentile is computed on a strided subsample.
                sub = arr[::8, ::8]
                hi = float(np.percentile(sub, 99.9))
                if hi <= 0.0:
                    hi = float(arr.max())
                scale = (255.0 / hi) if hi > 255.0 else 1.0
                arr = arr.astype(np.float32) * scale
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        # ---- Replicate single channel to RGB ----
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim != 3 or arr.shape[-1] < 3:
            raise RuntimeError(
                f"image at images['{image_key}'] has unsupported shape {arr.shape}."
            )
        arr = arr[:, :, :3]
        return arr

    def _he_array_axis_aligned(self, sdata, image_key='he',
                               element_name=None, sampling_distance=1.0,
                               target_um_per_px=None, pad=True,
                               revert_image=True):
        """Return an H&E ndarray axis-aligned in the global coordinate
        system **and cropped/padded so its pixel grid spans exactly the
        transcripts global bounding box**.

        ``_select_vectors`` zooms this array to the transcripts-derived
        VF grid and converts the resulting voxel indices to global µm
        via ``coordinate_offset + idx * sampling_distance``. That
        conversion is only correct if array index (row=0, col=0) maps
        to global (ty_min, tx_min) and (row=H, col=W) maps to
        (ty_max, tx_max). For 90°-multiple rotations the array is
        de-rotated losslessly with ``np.rot90`` and then cropped/padded
        to the transcripts bbox; for arbitrary rotations it falls back
        to ``spatialdata.rasterize`` at native pixel resolution onto
        the transcripts bbox (with the unavoidable interpolation blur).

        Parameters
        ----------
        target_um_per_px : float, optional
            Only used by the rasterize fallback (non-90° rotation case).
            If ``None``, the native pixel size is auto-detected.
        revert_image : bool, default True
            Background polarity of the staining image. ``True`` for
            bright-background / dark-tissue images (e.g. H&E); the bbox
            padding is filled with white (255) so it reads as background.
            ``False`` for dark-background / bright-tissue images (e.g.
            polyT); padding is filled with black (0) so it reads as
            background. Must match the value passed to the otsu / mask
            steps so padding is never mistaken for tissue.
        """
        from spatialdata.transformations import get_transformation as _get_tf
        from spatialdata import get_extent as _get_extent

        if image_key not in sdata.images:
            raise KeyError(
                f"sdata.images does not contain key '{image_key}'. "
                f"Available: {list(sdata.images.keys())}"
            )

        # Pull the underlying array at native H&E pixel resolution
        arr = self._he_array_from_sdata(sdata, image_key=image_key)  # (H, W, 3) uint8
        H, W = arr.shape[:2]

        # Inspect the affine: image-pixel → global
        try:
            tf = _get_tf(sdata.images[image_key], to_coordinate_system='global')
            M = np.asarray(tf.to_affine_matrix(
                input_axes=('x', 'y'), output_axes=('x', 'y')
            ))
        except Exception:
            # No transformation info: assume already axis-aligned and aligned
            # to the transcripts bbox. Best we can do.
            return arr

        A = M[:2, :2]
        t = M[:2, 2]

        # ---- Resolve transcripts global bbox (needed for crop) ----
        points_key = element_name
        if points_key is None:
            if 'transcripts' in sdata.points:
                points_key = 'transcripts'
            elif sdata.points:
                points_key = next(iter(sdata.points.keys()))

        tx_min = tx_max = ty_min = ty_max = None
        if points_key is not None:
            try:
                ext = _get_extent(sdata.points[points_key],
                                  coordinate_system='global')
                tx_min, tx_max = ext['x']
                ty_min, ty_max = ext['y']
            except Exception:
                pass

        # Detect the rotation angle of A (assumes rotation + isotropic scale).
        # A col 0 = scale * (cos angle, sin angle).
        angle_deg = float(np.degrees(np.arctan2(A[1, 0], A[0, 0])))

        # Snap to the nearest multiple of 90°
        nearest_k = int(round(angle_deg / 90.0)) % 4
        residual_deg = abs(angle_deg - nearest_k * 90.0)
        # Wrap residual into [-45, 45] for the comparison
        if residual_deg > 45:
            residual_deg = 90 - residual_deg

        if residual_deg < 5.0:
            # ---- Pure 90°-multiple rotation: undo it losslessly ----
            k_rot = (-nearest_k) % 4
            arr_rot = np.ascontiguousarray(np.rot90(arr, k=k_rot))

            # If we couldn't resolve the transcripts bbox, return uncropped
            # (legacy behaviour). Only happens when sdata has no points.
            if tx_min is None:
                return arr_rot

            # ---- Compute the de-rotated array's global extent via the
            # original image's 4 pixel corners ----
            corners_px = np.array(
                [[0.0, 0.0], [W, 0.0], [0.0, H], [W, H]], dtype=np.float64
            )
            corners_global = (A @ corners_px.T).T + t  # (4, 2)
            ix0 = float(corners_global[:, 0].min())
            iy0 = float(corners_global[:, 1].min())

            # Native isotropic pixel size in µm
            s = float(np.sqrt(abs(np.linalg.det(A))))
            if not np.isfinite(s) or s <= 0:
                return arr_rot

            Hr, Wr = arr_rot.shape[:2]

            # ---- Pixel indices in arr_rot that bound the transcripts bbox ----
            col_start = int(np.floor((tx_min - ix0) / s))
            col_end   = int(np.ceil ((tx_max - ix0) / s))
            row_start = int(np.floor((ty_min - iy0) / s))
            row_end   = int(np.ceil ((ty_max - iy0) / s))

            if not pad:
                # OTSU / content use (pad=False): return ONLY the real H&E
                # pixels inside the bbox window, clipped to the image — no
                # padding at all. Padding of ANY constant colour adds a large
                # artificial mode to the intensity histogram and biases the
                # Otsu threshold (black -> too high, white -> too low). Tiles
                # whose bbox exceeds the H&E simply contribute their real
                # pixels, matching the legacy raw-tif Otsu behaviour.
                vr0, vr1 = max(0, row_start), min(arr_rot.shape[0], row_end)
                vc0, vc1 = max(0, col_start), min(arr_rot.shape[1], col_end)
                return np.ascontiguousarray(arr_rot[vr0:vr1, vc0:vc1, :])

            # ---- Pad with white if transcripts bbox extends beyond H&E ----
            pad_left   = max(0, -col_start)
            pad_top    = max(0, -row_start)
            pad_right  = max(0, col_end - Wr)
            pad_bottom = max(0, row_end - Hr)
            if pad_left or pad_top or pad_right or pad_bottom:
                # Pad with the BACKGROUND colour so padding never masquerades
                # as tissue. revert_image=True (H&E): background is white, and
                # the otsu/mask path detects tissue via he_inv = 255 - gray, so
                # black padding (he_inv=255) would masquerade as max foreground
                # and inflate the Otsu threshold. revert_image=False (polyT):
                # background is dark, the mask uses he_inv = gray directly, so
                # pad with black (0) for the same reason.
                pad_value = 255 if revert_image else 0
                arr_rot = np.pad(
                    arr_rot,
                    ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                    mode='constant', constant_values=pad_value,
                )
                col_start += pad_left
                col_end   += pad_left
                row_start += pad_top
                row_end   += pad_top

            cropped = arr_rot[row_start:row_end, col_start:col_end, :]
            return np.ascontiguousarray(cropped)

        # ---- Fallback: arbitrary rotation → rasterize onto transcripts bbox ----
        from spatialdata import rasterize as _rasterize

        if tx_min is None:
            raise RuntimeError(
                "sdata has no points element; cannot derive H&E bbox for "
                "non-90° rotated affine."
            )

        if target_um_per_px is None:
            target_um_per_px = float(np.sqrt(abs(np.linalg.det(A))))
            if not np.isfinite(target_um_per_px) or target_um_per_px <= 0:
                target_um_per_px = float(sampling_distance)
        target_unit_to_pixels = 1.0 / float(target_um_per_px)

        raster = _rasterize(
            sdata.images[image_key],
            axes=('x', 'y'),
            min_coordinate=[tx_min, ty_min],
            max_coordinate=[tx_max, ty_max],
            target_coordinate_system='global',
            target_unit_to_pixels=target_unit_to_pixels,
        )
        arr = raster.values if hasattr(raster, 'values') else np.asarray(raster)
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (3, 4):
            arr = np.moveaxis(arr, 0, -1)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        arr = arr[:, :, :3]
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.0:
                arr = (arr * 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    @staticmethod
    def _rgb_to_intensity(img, intensity='luminance'):
        """Collapse an (H, W, 3) RGB image to a single-channel intensity.

        ``intensity='luminance'`` uses the BT.601 perceptual weighting
        (``0.299 R + 0.587 G + 0.114 B``) — correct for brightfield / H&E
        photographs where the channels are correlated. ``intensity='max'``
        uses the per-pixel channel maximum — correct for fluorescence
        composite images (e.g. CosMx CellComposite), where the tissue
        signal may live almost entirely in one channel (e.g. blue DAPI
        nuclei) and the luminance weighting would crush it (blue is only
        weighted 0.114, so a blue-dominant composite would be discarded).

        Parameters
        ----------
        img : ndarray (H, W, 3)
        intensity : {'luminance', 'max'}, default 'luminance'

        Returns
        -------
        gray : ndarray (H, W) float
        """
        rgb = img[:, :, :3].astype(np.float64)
        if intensity == 'max':
            return rgb.max(axis=-1)
        if intensity == 'luminance':
            return (0.299 * rgb[:, :, 0]
                    + 0.587 * rgb[:, :, 1]
                    + 0.114 * rgb[:, :, 2])
        raise ValueError(
            f"intensity must be 'luminance' or 'max', got {intensity!r}.")

    def _otsu_thresholding(self, he_imgs, revert_image=True,
                           intensity='luminance'):
        """
        Compute per-image Otsu threshold and return the median.

        The threshold is computed on a tissue-bright intensity image so the
        downstream mask (``_select_vectors``) keeps high-signal pixels.
        ``revert_image=True`` inverts a bright-background image (H&E) to make
        the tissue bright; ``revert_image=False`` uses the grayscale directly
        for an already tissue-bright image (polyT). The same ``revert_image``
        MUST be used in :meth:`_select_vectors` so the threshold and mask
        operate on the same representation.

        Parameters
        ----------
        he_imgs : list of ndarray (H, W, 3)  RGB images
        revert_image : bool, default True
            ``True``  → invert (bright background, dark tissue; e.g. H&E).
            ``False`` → no inversion (dark background, bright tissue; e.g. polyT).
        intensity : {'luminance', 'max'}, default 'luminance'
            How RGB is collapsed to intensity (see :meth:`_rgb_to_intensity`).
            ``'luminance'`` for brightfield/H&E; ``'max'`` for fluorescence
            composites (e.g. CosMx CellComposite) so a single-channel signal
            such as blue DAPI is not crushed. MUST match the value used in
            :meth:`_select_vectors`.

        Returns
        -------
        median_otsu : float
        otsu_vals   : list of float  (per-image values)
        """
        import cv2

        otsu_vals = []
        for img in he_imgs:
            gray = self._rgb_to_intensity(img, intensity=intensity)
            inv = ((255.0 - gray) if revert_image else gray).astype(np.uint8)
            val, _ = cv2.threshold(inv, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            otsu_vals.append(float(val))
        return float(np.median(otsu_vals)), otsu_vals

    def _select_vectors(self, he_img, sdata, otsu_threshold,
                        element_name=None,
                        bandwidth=2.5, sampling_distance=1.0,
                        prune_coefficient=4.3, size_after_normalization=10,
                        log_transform=True, n_samples=5, block_um=10,
                        seed=42, filter_is_gene=True, genes=None,
                        revert_image=True, intensity='luminance',
):
        """
        Mask → position selection → KDE → normalization for a single image/sdata.

        Parameters
        ----------
        he_img        : ndarray (H, W, 3)  RGB image
        sdata         : spatialdata.SpatialData
        otsu_threshold: float  threshold to apply (from _otsu_thresholding)

        Returns
        -------
        gx_he, gy_he, gz_he : ndarray  voxel coords (local to this sdata)
        normalized_vectors  : ndarray (N, n_genes)
        vf_norm             : ndarray (N,)
        genes               : list[str]
        coordinate_offset   : list[float]  [x_min, y_min, z_min] in µm
        sample_stats        : dict
        """
        import cv2
        from scipy.ndimage import zoom
        from scipy.spatial import cKDTree
        from spatialdata.transformations import get_transformation as _get_tf

        H_he, W_he = he_img.shape[:2]

        # ---- 1. H&E → binary mask ----
        # he_inv is the tissue-bright intensity. revert_image=True inverts a
        # bright-background image (H&E); revert_image=False uses the grayscale
        # directly for an already tissue-bright image (polyT). Must match the
        # revert_image / intensity passed to _otsu_thresholding for the
        # threshold to apply to the same representation.
        he_gray = self._rgb_to_intensity(he_img, intensity=intensity)
        he_inv = (255.0 - he_gray) if revert_image else he_gray
        he_sub = np.clip(he_inv - otsu_threshold, 0.0, None)
        he_mask = (np.log1p(he_sub) > 0).astype(np.uint8)
        n_mask_pixels = int(he_mask.sum())

        # ---- 2. VF grid shape ----
        if getattr(self.dataset, 'shape', None) is None:
            self._prepare_grid(
                sdata,
                element_name=element_name,
                bandwidth=bandwidth,
                sampling_distance=sampling_distance,
                prune_coefficient=prune_coefficient,
                size_after_normalization=size_after_normalization,
            )
        vx, vy, vz = self.dataset.shape

        # ---- 3. Rescale mask to VF grid ----
        he_grid = zoom(he_mask, (vy / H_he, vx / W_he), order=0) > 0

        # ---- 4. Parse transcripts ----
        meta = self._parse_transcripts(
            sdata, sampling_distance=sampling_distance,
            filter_is_gene=filter_is_gene
        )

        ndim = 2 if vz == 1 else 3
        expression_threshold = 1 / (np.sqrt(2 * np.pi) * bandwidth) ** ndim
        norm_threshold = expression_threshold * 2

        # ---- 5. Pixel-proportional sampling ----
        x_scale = y_scale = None
        for _etype, _ename, _elem in sdata.gen_elements():
            if _etype == 'points':
                _sc = _get_tf(_elem, get_all=True)
                _tf = _sc.get('global', None)
                if _tf is not None and hasattr(_tf, 'scale'):
                    x_scale = _tf.scale[0]
                    y_scale = _tf.scale[1]
                else:
                    # Identity or affine without scale attr: assume scale=1
                    try:
                        _aff = _tf.to_affine_matrix(input_axes=('x', 'y'), output_axes=('x', 'y'))
                        x_scale = abs(_aff[0, 0]) if _aff.shape[0] > 0 else 1.0
                        y_scale = abs(_aff[1, 1]) if _aff.shape[0] > 1 else 1.0
                    except Exception:
                        x_scale = 1.0
                        y_scale = 1.0
                break
        if x_scale is None:
            raise RuntimeError('No points element found in sdata.')

        ys_all, xs_all = np.where(he_grid)
        block_px = (block_um * x_scale) * (block_um * y_scale)
        n_select = min(int(n_mask_pixels / block_px * n_samples), len(ys_all))
        if n_select == 0:
            # Blank / near-blank tile: nothing to sample. Return empty arrays
            # so the chunked driver can skip this tile gracefully. The
            # non-chunked driver inspects ``sample_stats['n_select']`` and
            # surfaces an informative error to the user.
            empty_int = np.empty(0, dtype=np.int64)
            empty_nvecs = np.empty((0, 0), dtype=np.float64)
            empty_vf = np.empty(0, dtype=np.float64)
            sample_stats = {
                'n_mask_pixels': n_mask_pixels,
                'block_px': float(block_px),
                'n_select': 0,
                'candidate_pixels': int(he_grid.sum()),
                'dropped_xy_without_transcript_support': 0,
                'oversampled_count': 0,
                'kept_count': 0,
                'norm_threshold': float(norm_threshold),
            }
            return (empty_int, empty_int, empty_int, empty_nvecs, empty_vf,
                    [], meta['coordinate_offset'], sample_stats)

        rng = np.random.default_rng(seed)
        idx = rng.choice(len(ys_all), n_select, replace=False)
        gx_cand = xs_all[idx].astype(np.int64)
        gy_cand = ys_all[idx].astype(np.int64)

        rp = int(np.floor((bandwidth * prune_coefficient) / sampling_distance + 1e-12))

        dropped_xy = 0
        if not meta['is_3d'] or vz == 1:
            gx_raw = gx_cand
            gy_raw = gy_cand
            gz_raw = np.zeros(len(gx_cand), dtype=np.int64)
        else:
            xy_grid_coords = np.column_stack([meta['xc_g'], meta['yc_g']]).astype(np.float64)
            xy_tree = cKDTree(xy_grid_coords)
            gx_list, gy_list, gz_list = [], [], []
            for x, y in zip(gx_cand, gy_cand):
                neighbors = xy_tree.query_ball_point([float(x), float(y)], r=rp, p=np.inf)
                if len(neighbors) == 0:
                    dropped_xy += 1
                    continue
                z_vals = meta['zc_g'][np.asarray(neighbors, dtype=np.int64)]
                z_vals = z_vals[(z_vals >= 0) & (z_vals < vz)]
                if z_vals.size == 0:
                    dropped_xy += 1
                    continue
                gx_list.append(x)
                gy_list.append(y)
                gz_list.append(vz // 2)
            if len(gx_list) == 0:
                raise RuntimeError('No sampled voxels had nearby transcript support.')
            gx_raw = np.asarray(gx_list, dtype=np.int64)
            gy_raw = np.asarray(gy_list, dtype=np.int64)
            gz_raw = np.asarray(gz_list, dtype=np.int64)

        coords = np.unique(np.stack([gx_raw, gy_raw, gz_raw], axis=1), axis=0)
        gx_raw, gy_raw, gz_raw = coords[:, 0], coords[:, 1], coords[:, 2]

        # ---- 6. KDE + normalization ----
        query_voxels = np.stack([gx_raw, gy_raw, gz_raw], axis=1)
        normalized_vectors, vf_norm, genes, coordinate_offset = \
            self._compute_normalized_vectors(
                query_voxels, meta,
                genes=genes,
                bandwidth=bandwidth,
                sampling_distance=sampling_distance,
                prune_coefficient=prune_coefficient,
                size_after_normalization=size_after_normalization,
                log_transform=log_transform,
            )

        # ---- 7. Filter by norm threshold ----
        keep = vf_norm > norm_threshold
        gx_he = gx_raw[keep]
        gy_he = gy_raw[keep]
        gz_he = gz_raw[keep]
        normalized_vectors = normalized_vectors[keep]
        vf_norm = vf_norm[keep]

        sample_stats = {
            'n_mask_pixels': n_mask_pixels,
            'block_px': float(block_px),
            'n_select': n_select,
            'candidate_pixels': int(he_grid.sum()),
            'dropped_xy_without_transcript_support': int(dropped_xy),
            'oversampled_count': len(gx_raw),
            'kept_count': int(keep.sum()),
            'norm_threshold': float(norm_threshold),
        }

        return gx_he, gy_he, gz_he, normalized_vectors, vf_norm, genes, coordinate_offset, sample_stats

    def compute_vf(self, grid, *, bin_um=2.0, sampling_distance=None, panel=None,
                   sigma_bin=None, bandwidth_um=None, prune_coefficient=4.3,
                   size_after_normalization=10.0, device='cuda', chunk_h=64,
                   chunk_w=512, store_chunk=32, mode='auto',
                   log_transform=True):
        """Build the count-data vector field (GPU blur) and persist it to zarr.

        This is the COUNT-platform entry point (Stereo-seq / Visium HD /
        StrataMap): it rasterizes bin counts and Gaussian-blurs them into
        ``vf_normalized`` + ``vf_norm``, writing only the normalized field.
        Transcript platforms do NOT call this — their field is computed per-chunk
        inside :meth:`map_celltypes`.

        Two independent resolutions (see :func:`cellman.spatial._count.compute_vf_blur`):

        Parameters
        ----------
        grid : GridDataset or AnnData
            A :class:`cellman.spatial.GridDataset` (raw counts are aggregated to a
            ``bin_um`` grid using ``grid.origin``), or a pre-aggregated bin-level
            AnnData with ``obs['array_row'/'array_col']``.
        bin_um : float
            **Blur raster** bin size in µm — the grid the Gaussian is computed on
            (finer = more accurate blur). Ignored if ``grid`` is already gridded.
        sampling_distance : float, optional
            **Output vector-field** grid spacing in µm. The blurred raster is
            decimated by ``round(sampling_distance / bin_um)`` so the stored vf can
            be coarser than the blur raster (fewer voxels on disk). Defaults to
            ``bin_um`` (output == raster).
        bandwidth_um : float, optional
            Gaussian bandwidth (σ) in **µm** — the intuitive knob. Converted to
            bin units internally (``sigma_bin = bandwidth_um / bin_um``). Give
            EITHER ``bandwidth_um`` OR ``sigma_bin`` (bin units); ``bandwidth_um``
            wins if both. Default: ``sigma_bin=1`` (bandwidth = ``bin_um`` µm).
        prune_coefficient : float, default 4.3
            Gaussian kernel truncation radius in σ units: ``radius = ceil(
            prune_coefficient * sigma_bin)`` bins (= ``prune_coefficient *
            bandwidth`` µm). Matches SSAM's default 4.3 (radius ≈ cell diameter);
            lower (e.g. 3.0 ≈ 99.7% mass) truncates tighter and runs faster.
        panel : list[str], optional
            Gene panel to restrict to before rasterization.
        chunk_h, chunk_w : int
            GPU **compute** tile (output rows × cols per blur pass). Large = fewer
            kernel launches / less halo recomputation. Does NOT set the on-disk chunk.
        store_chunk : int or (h, w), default 32
            On-disk chunk shape of ``vf_normalized`` (gene axis always full). Small
            SQUARE tiles let a sparse ``downsample_vector`` gather decode only the
            tiles containing sampled voxels instead of full-width row bands. Keep
            ``chunk_h``/``chunk_w`` integer multiples of it (defaults are). ``None``
            reverts to the legacy store == compute tile.
        """
        from ._count import compute_vf_blur
        from ._griddataset import GridDataset
        from ._io import aggregate_to_grid

        # unified input: a count SpatialData (tables['table'] + images['staining'])
        # is adapted to the lightweight GridDataset view the backend consumes. The
        # sdata itself is remembered so the final celltype map can be written back.
        try:
            from spatialdata import SpatialData as _SD
        except Exception:  # noqa: BLE001
            _SD = ()
        if _SD and isinstance(grid, _SD):
            # regime detection: count SpatialData → GridDataset.from_sdata succeeds;
            # a transcript/imaging SpatialData (Xenium/MERFISH) cannot be gridded →
            # route to the KDE field PRODUCER (`_transcript_gpu`, do_map=False) which
            # streams the field to a tensorstore writer. `mode` forces the branch.
            _gd = None
            _is_transcript = (mode == 'transcript')
            if mode == 'auto':
                try:
                    _gd = GridDataset.from_sdata(grid)
                except Exception:  # noqa: BLE001
                    _is_transcript = True
            if _is_transcript:
                bw = float(bandwidth_um) if bandwidth_um is not None else 2.5
                self._transcript_gpu(
                    grid, do_map=False, save_vf=True,
                    bandwidth=bw,
                    sampling_distance=(1.0 if sampling_distance is None
                                       else float(sampling_distance)),
                    prune_coefficient=prune_coefficient,
                    size_after_normalization=size_after_normalization,
                    log_transform=log_transform, gpu_device=device)
                return None
            self._count_sdata = grid
            self.dataset._input_data_type = 'spatialdata'
            self.dataset._original_input_data = grid
            grid = _gd if _gd is not None else GridDataset.from_sdata(grid)

        if isinstance(grid, GridDataset):
            adata = grid.raw if panel is None else grid.raw[:, [g for g in panel
                                                    if g in set(grid.raw.var_names)]]
            bin_adata = aggregate_to_grid(adata, bin_um, origin=grid.origin)
            self._count_grid = grid          # remembered for downsample_vector
        else:
            bin_adata = grid if panel is None else grid[:, [g for g in panel
                                                    if g in set(grid.var_names)]]
            bin_um = float(bin_adata.uns.get("bin_size_um", bin_um))
        # bandwidth: prefer the µm knob; else sigma_bin (bins); else default σ=1 bin.
        if bandwidth_um is not None:
            sigma_bin = float(bandwidth_um) / float(bin_um)
        elif sigma_bin is None:
            sigma_bin = 1.0
        compute_vf_blur(self, bin_adata, sigma_bin=sigma_bin, chunk_h=chunk_h,
                        chunk_w=chunk_w, store_chunk=store_chunk,
                        bin_size_um=bin_um, sampling_distance=sampling_distance,
                        prune_coefficient=prune_coefficient,
                        size_after_normalization=size_after_normalization,
                        device=device)
        return bin_adata

    def downsample_vector(self, sdata=None, *, grid=None,
                          auto_gpu=True, gpu_device='cuda', **kwargs):
        """Public entry point with automatic GPU/CPU dispatch.

        Returns an :class:`anndata.AnnData` of the sampled voxels (``X`` =
        normalized_vectors, ``obsm['spatial']``, ``obs['vf_norm']``, ``var`` =
        genes) — feed it to ``sc.pp.pca`` / ``sc.pp.neighbors`` and then
        :meth:`cluster_vectors`::

            adata = analysis.downsample_vector(sdata, ...)
            sc.pp.pca(adata); sc.pp.neighbors(adata)
            analysis.cluster_vectors(adata)   # writes adata.obs['leiden'] in place


        Two modes, auto-detected:

        * **count** — when ``grid`` is given (or omitted but a ``compute_vf`` grid
          was remembered) AND the store already holds a blur ``vf_normalized``:
          voxels are sampled from the tissue image and *looked up* in that field
          (no KDE). Delegates to
          :func:`cellman.spatial._count.select_vectors_from_image`.
        * **transcript** — otherwise: voxels are sampled from the H&E tissue mask
          and KDE is computed at each (the original path below).


        If ``auto_gpu`` is True (default) and a usable CUDA device is
        detected (``torch + triton + cuda.is_available()``), the underlying
        ``ssam.utils.kde_at_points`` is transparently routed to the Triton
        GPU kernel for the duration of this call. Otherwise the original
        C++/OpenMP CPU implementation is used. The user therefore does
        not need to wrap the call in ``with gpu_kde(...): ...``.

        All other arguments are forwarded verbatim to
        :meth:`_downsample_vector_impl` (see its docstring).

        Parameters
        ----------
        sdata : spatialdata.SpatialData, optional
            Forwarded to the implementation (used when ``chunked=False``).
        auto_gpu : bool, default True
            If False, force CPU regardless of CUDA availability.
        gpu_device : str, default ``'cuda'``
            Torch device string, e.g. ``'cuda'``, ``'cuda:0'``, ``'cuda:1'``.
        **kwargs
            All other ``downsample_vector`` parameters (chunked, sdata_dir,
            n_samples, block_um, bandwidth, sampling_distance,
            revert_image, ...).

            revert_image : bool, default True
                Background polarity of the staining image used for the tissue
                mask. ``True`` for bright-background / dark-tissue images such
                as H&E (the image is inverted before Otsu thresholding).
                ``False`` for dark-background / bright-tissue images such as
                polyT (the grayscale is thresholded directly, no inversion).
            intensity : {'luminance', 'max'}, default 'luminance'
                How the RGB staining image is collapsed to a single-channel
                intensity for the tissue mask. ``'luminance'`` (BT.601
                weighting) suits brightfield / H&E photographs. ``'max'``
                (per-pixel channel maximum) suits fluorescence composites
                such as CosMx CellComposite, where the tissue signal can sit
                almost entirely in one channel (e.g. blue DAPI nuclei) and
                luminance would crush it (blue weighted only 0.114).
        """
        # ---- count path: sample voxels from the image and look them up in the
        #      precomputed blur field (no KDE) ----
        grid = grid if grid is not None else getattr(self, '_count_grid', None)
        zg = self.dataset.zarr_group
        is_blur = ('vf_normalized' in zg) and (zg.attrs.get('backend') == 'blur')
        # a count SpatialData passed positionally (or a fresh kernel with no
        # remembered grid) → adapt to the GridDataset view the lookup needs.
        if grid is None and is_blur and sdata is not None:
            try:
                from spatialdata import SpatialData as _SD
            except Exception:  # noqa: BLE001
                _SD = ()
            if _SD and isinstance(sdata, _SD) and len(getattr(sdata, 'tables', {})):
                from ._griddataset import GridDataset
                grid = GridDataset.from_sdata(sdata)
                self._count_sdata = sdata
                self.dataset._input_data_type = 'spatialdata'
                self.dataset._original_input_data = sdata
        if grid is not None and is_blur:
            from ._count import select_vectors_from_image
            # the vf grid is at the OUTPUT spacing (sampling_distance), not the
            # blur raster (bin_size_um) — map image px → µm → output bin by it.
            grid_um = float(zg.attrs.get('sampling_distance',
                            zg.attrs.get('bin_size_um', 1.0)))
            sel_keys = ('n_samples_per_block', 'block_um', 'otsu_threshold',
                        'norm_threshold_mult', 'vf_norm_percentile',
                        'lookup_chunk_h', 'seed',
                        'lookup_n_workers', 'lookup_ram_gb')
            sel_kwargs = {k: kwargs[k] for k in sel_keys if k in kwargs}
            # accept the transcript alias n_samples -> n_samples_per_block
            if 'n_samples' in kwargs and 'n_samples_per_block' not in sel_kwargs:
                sel_kwargs['n_samples_per_block'] = kwargs['n_samples']
            select_vectors_from_image(self, grid, bin_size_um=grid_um, **sel_kwargs)
            return self._downsampled_adata()

        # ---- transcript path (original KDE-at-sampled-points) ----
        from contextlib import nullcontext as _nullcontext
        if auto_gpu:
            try:
                from ._kde_gpu import auto_kde_context as _auto_ctx
                _ctx = _auto_ctx(gpu_device)
            except ImportError:
                _ctx = _nullcontext()
        else:
            _ctx = _nullcontext()
        with _ctx:
            self._downsample_stats = self._downsample_vector_impl(sdata=sdata, **kwargs)
        return self._downsampled_adata()

    def _downsample_vector_impl(self, sdata=None, element_name=None,
                         bandwidth=2.5, sampling_distance=1.0, prune_coefficient=4.3,
                         size_after_normalization=10, log_transform=True,
                         n_samples=5, block_um=10, seed=42,
                         filter_is_gene=True, genes=None,
                         chunked=False, sdata_dir=None,
                         image_key='he', otsu_threshold=None,
                         revert_image=True, intensity='luminance'):
        """
        Select representative vectors from H&E-intensity-guided positions.

        Voxels are sampled uniformly at random from the H&E tissue mask,
        with the number of samples proportional to the mask area:
        ``n_select = int(n_mask_pixels / block_px * n_samples)``
        where ``block_px = (block_um * x_scale) * (block_um * y_scale)``.

        The H&E image is now read directly from ``sdata.images[image_key]``;
        the caller is responsible for ensuring it has been cropped to the
        sequenced region and rasterized onto a grid that spatially
        corresponds to the morphology image.

        Stores results in ``ds.downsampled_vectors``, ``ds.normalized_vectors``,
        ``ds.genes``, ``ds.coordinate_offset``, and persists them to zarr.

        Parameters
        ----------
        sdata : spatialdata.SpatialData
            Used when ``chunked=False``. Must contain ``images[image_key]``.
        sdata_dir : str or pathlib.Path
            Used when ``chunked=True``. Directory containing one or more
            ``*.zarr`` SpatialData stores; each must contain ``images[image_key]``.
        image_key : str, default 'he'
            Key under which the staining / morphology image (e.g. H&E or
            polyT) is stored in ``sdata.images``.
        n_samples : int
            Number of voxels to sample per ``block_um × block_um`` µm block.
        block_um : float
            Block size in µm used to determine sampling density.
        revert_image : bool, default True
            Background polarity of the staining image used to build the
            tissue mask (Otsu + mask + bbox padding).
            ``True``  → bright background, dark tissue (e.g. H&E); the image
            is inverted (``255 - gray``) before thresholding.
            ``False`` → dark background, bright tissue (e.g. polyT); the
            grayscale is used directly with no inversion.
        intensity : {'luminance', 'max'}, default 'luminance'
            How the RGB staining image is collapsed to intensity for the
            tissue mask. ``'luminance'`` (BT.601) for brightfield / H&E;
            ``'max'`` (per-pixel channel maximum) for fluorescence composites
            (e.g. CosMx CellComposite) so a single-channel signal such as blue
            DAPI is not crushed by the luminance weighting.

        Returns
        -------
        sample_stats : dict
        """
        from pathlib import Path as _Path
        import spatialdata as _sd

        helper_kwargs = dict(
            element_name=element_name,
            bandwidth=bandwidth, sampling_distance=sampling_distance,
            prune_coefficient=prune_coefficient,
            size_after_normalization=size_after_normalization,
            log_transform=log_transform,
            n_samples=n_samples, block_um=block_um, seed=seed,
            filter_is_gene=filter_is_gene, genes=genes,
            revert_image=revert_image, intensity=intensity,
        )

        if not chunked:
            # ================================================================
            # chunked=False : sdata 하나 (H&E는 sdata.images[image_key])
            # ================================================================
            if sdata is None:
                raise ValueError('chunked=False requires sdata.')

            he_img = self._he_array_axis_aligned(
                sdata, image_key=image_key,
                element_name=element_name, sampling_distance=sampling_distance,
                revert_image=revert_image,
            )
            if otsu_threshold is None:
                # Otsu on the un-padded real H&E pixels (pad=False); the
                # background padding in he_img is only for mask/grid alignment.
                otsu_img = self._he_array_axis_aligned(
                    sdata, image_key=image_key,
                    element_name=element_name, sampling_distance=sampling_distance,
                    pad=False, revert_image=revert_image,
                )
                otsu_threshold, _ = self._otsu_thresholding(
                    [otsu_img], revert_image=revert_image, intensity=intensity)

            # _select_vectors 호출 (ds.shape 초기화 포함)
            self.dataset.shape = None
            gx_he, gy_he, gz_he, normalized_vectors, vf_norm_all, genes, coordinate_offset, sample_stats = \
                self._select_vectors(he_img, sdata, otsu_threshold, **helper_kwargs)

            if sample_stats.get('n_select', 0) == 0:
                raise RuntimeError(
                    f"n_select is 0 (n_mask_pixels="
                    f"{sample_stats.get('n_mask_pixels', 0)}, "
                    f"otsu_threshold={otsu_threshold}). "
                    f"The H&E mask is empty after thresholding. Check "
                    f"n_samples / block_um, or verify that images[{image_key!r}] "
                    f"contains tissue signal."
                )

            if self.verbose:
                vx, vy, vz = self.dataset.shape
                print(f'downsample_vector: vf grid ({vx}, {vy}, {vz})')
                print(f'  otsu_threshold   : {otsu_threshold}')
                for k in ('n_mask_pixels', 'block_px', 'n_select',
                          'dropped_xy_without_transcript_support', 'kept_count'):
                    print(f'  {k:40s}: {sample_stats[k]}')

        else:
            # ================================================================
            # chunked=True : sdata_dir의 *.zarr 각각에서 images[image_key] 사용
            # ================================================================
            if sdata_dir is None:
                raise ValueError('chunked=True requires sdata_dir.')

            sdata_dir = _Path(sdata_dir)
            zarr_paths = sorted(sdata_dir.glob('*.zarr'))
            if not zarr_paths:
                raise RuntimeError(f'No *.zarr stores found in {sdata_dir}.')

            if self.verbose:
                print(f'downsample_vector (chunked): {len(zarr_paths)} sdata stores found.')

            # ---- Step 1: 청크별 Otsu (streaming, 한 번에 한 H&E만 RAM에) ----
            # Otsu는 grayscale 강도 히스토그램에서 임계 스칼라 하나만 뽑는 통계라
            #   (a) 회전(np.rot90)은 히스토그램 불변 → 불필요,
            #   (b) transcripts bbox crop도 불필요 — 그런데 _he_array_axis_aligned는
            #       crop을 위해 get_extent(sdata.points)로 수백만 transcript를 전부
            #       스캔(타일당 ~140s)한다. 이게 otsu 단계의 사실상 전부의 비용이었다.
            # → raw H&E를 그대로 읽어(≈1s) stride 서브샘플 후 otsu. 임계는 마스킹 단계의
            #   풀해상도 이미지(pad=True)에 적용되므로 마스크 해상도는 그대로 유지된다.
            # 검증(Xenium Prime mouse brain, 6 tiles): stride=4 raw otsu가 axis-aligned
            #   pad=False otsu와 타일별로 동일(median 79.0). ~7min → ~20s.
            _OTSU_STRIDE = 4
            otsu_vals = []
            for zp in zarr_paths:
                s = _sd.read_zarr(zp)
                arr = self._he_array_from_sdata(s, image_key=image_key)   # (H, W, 3) raw, no crop/rot
                otsu_img = arr[::_OTSU_STRIDE, ::_OTSU_STRIDE, :]
                _, single_vals = self._otsu_thresholding(
                    [otsu_img], revert_image=revert_image, intensity=intensity)
                otsu_vals.append(single_vals[0])
                del s, arr, otsu_img
            otsu_threshold = float(np.median(otsu_vals))

            if self.verbose:
                for zp, v in zip(zarr_paths, otsu_vals):
                    print(f'  {zp.stem}: otsu={v:.0f}')
                print(f'  median otsu: {otsu_threshold}')

            # ---- Step 2: 청크별 select_vectors 실행 (streaming + GPU 캐시 정리) ----
            import gc as _gc
            try:
                import torch as _torch
                _has_cuda = _torch.cuda.is_available()
            except ImportError:
                _torch, _has_cuda = None, False

            all_gx, all_gy, all_gz = [], [], []
            all_nvecs = []
            all_vfnorm = []     # per-vector vf_norm (local KDE density) per chunk
            all_chunk_genes = []
            all_offsets = []    # coordinate_offset per chunk ([x_min, y_min, z_min] in µm)
            all_stats = []

            for zp in zarr_paths:
                if self.verbose:
                    print(f'\n  [{zp.stem}]')

                chunk_sdata = _sd.read_zarr(zp)
                he_img = self._he_array_axis_aligned(
                    chunk_sdata, image_key=image_key,
                    element_name=element_name, sampling_distance=sampling_distance,
                    revert_image=revert_image,
                )

                # 각 청크마다 ds.shape를 초기화해서 _prepare_grid가 재실행되도록 함
                self.dataset.shape = None
                gx, gy, gz, nvecs, vfn, chunk_genes, coord_off, stats = \
                    self._select_vectors(he_img, chunk_sdata, otsu_threshold, **helper_kwargs)

                # Blank tile (no foreground above otsu): skip without contributing
                # to the global offset / gene-union / concatenation.
                if stats.get('n_select', 1) == 0:
                    if self.verbose:
                        print(f'    skipped (blank tile, n_mask_pixels='
                              f'{stats.get("n_mask_pixels", 0)})')
                    del he_img, chunk_sdata, gx, gy, gz, nvecs, vfn
                    _gc.collect()
                    if _has_cuda:
                        _torch.cuda.empty_cache()
                    continue

                # # --- 수정 후 삭제됨 ---
                # if all_genes is None:
                #     all_genes = chunk_genes
                # elif all_genes != chunk_genes:
                #     raise RuntimeError(
                #         f'Gene list mismatch between chunks. all_genes has {hp.stem} genes, '
                #         f'but chunk {hp.stem} has {chunk_genes}.'
                #     )

                all_gx.append(gx)
                all_gy.append(gy)
                all_gz.append(gz)
                all_nvecs.append(nvecs)
                all_vfnorm.append(vfn)          # row-aligned with gx/gy/gz/nvecs
                all_chunk_genes.append(chunk_genes)
                all_offsets.append(coord_off)   # [x_min, y_min, z_min] in µm
                all_stats.append(stats)

                if self.verbose:
                    print(f'    kept: {stats["kept_count"]:,}  offset: {coord_off}')

                # ---- per-chunk cleanup: free large H&E array, sdata refs, and GPU cache ----
                del he_img, chunk_sdata
                _gc.collect()
                if _has_cuda:
                    _torch.cuda.empty_cache()

            # All tiles were blank — nothing to assemble.
            if not all_offsets:
                raise RuntimeError(
                    f'All {len(zarr_paths)} chunks were blank after otsu '
                    f'thresholding (otsu_threshold={otsu_threshold}). No '
                    f'vectors were sampled. Check the H&E key '
                    f'({image_key!r}) and whether the image contains signal.'
                )

            # ---- Step 3: local voxel 좌표 → global voxel 좌표 ----
            # global origin = 전체 청크 중 가장 작은 x_min, y_min, z_min
            global_x_min = min(off[0] for off in all_offsets)
            global_y_min = min(off[1] for off in all_offsets)
            global_z_min = min(off[2] for off in all_offsets)
            coordinate_offset = [global_x_min, global_y_min, global_z_min]

            gx_he_list, gy_he_list, gz_he_list = [], [], []
            for gx, gy, gz, off in zip(all_gx, all_gy, all_gz, all_offsets):
                dx = round((off[0] - global_x_min) / sampling_distance)
                dy = round((off[1] - global_y_min) / sampling_distance)
                dz = round((off[2] - global_z_min) / sampling_distance)
                gx_he_list.append(gx + dx)
                gy_he_list.append(gy + dy)
                gz_he_list.append(gz + dz)

            gx_he = np.concatenate(gx_he_list)
            gy_he = np.concatenate(gy_he_list)
            gz_he = np.concatenate(gz_he_list)

            # Correct the global coordinate metadata. Each chunk's _prepare_grid
            # persisted *its own* offset/bounds to zarr, so the LAST chunk's
            # (single-tile) values were left behind — wrong for the merged grid.
            # Recompute the global extent from the merged grid so that the saved
            # metadata, a reload from zarr, and every consumer of
            # original_coordinate_bounds (plot extents, etc.) see the full map.
            gx_max_um = global_x_min + float(gx_he.max()) * sampling_distance
            gy_max_um = global_y_min + float(gy_he.max()) * sampling_distance
            gz_max_um = global_z_min + float(gz_he.max()) * sampling_distance
            self.dataset.coordinate_bounds = [
                gx_max_um - global_x_min,
                gy_max_um - global_y_min,
                gz_max_um - global_z_min,
            ]
            self.dataset.original_coordinate_bounds = [
                [global_x_min, gx_max_um],
                [global_y_min, gy_max_um],
                [global_z_min, gz_max_um],
            ]

            # --- 수정 후 삭제됨 ---
            # normalized_vectors = np.concatenate(all_nvecs, axis=0)
            # genes = all_genes
            # ---  

            # --- 수정 후 추가됨 ---
            # ---- Align nvecs across chunks via union of gene panels ----
            # Different chunks (especially sparse edge tiles) may observe
            # different subsets of the panel. Take the union, sorted, and
            # zero-pad each chunk's nvecs to the union columns. This is
            # mathematically identical to running with the union panel from
            # the start, because missing genes contribute 0 to both KDE
            # values and the L1 sum used for normalization.
            genes = sorted(set().union(*all_chunk_genes))
            gene_to_idx = {g: i for i, g in enumerate(genes)}
            n_union = len(genes)
            if self.verbose:
                per_chunk_sizes = [len(g) for g in all_chunk_genes]
                if min(per_chunk_sizes) != max(per_chunk_sizes) or max(per_chunk_sizes) != n_union:
                    print(f'  gene panel union: {n_union} '
                          f'(per-chunk min={min(per_chunk_sizes)}, max={max(per_chunk_sizes)})')

            aligned_nvecs = []
            for nvecs, chunk_genes in zip(all_nvecs, all_chunk_genes):
                if list(chunk_genes) == genes:
                    aligned_nvecs.append(nvecs)
                else:
                    aligned = np.zeros((nvecs.shape[0], n_union), dtype=nvecs.dtype)
                    col_idx = np.fromiter(
                        (gene_to_idx[g] for g in chunk_genes),
                        dtype=np.int64, count=len(chunk_genes),
                    )
                    aligned[:, col_idx] = nvecs
                    aligned_nvecs.append(aligned)
            normalized_vectors = np.concatenate(aligned_nvecs, axis=0)
            # vf_norm is per-row (independent of gene panel), concatenated in the
            # same chunk order as gx_he / normalized_vectors → row-aligned.
            vf_norm_all = np.concatenate(all_vfnorm)
            # ---

            sample_stats = {
                'n_chunks': len(zarr_paths),
                'otsu_threshold': otsu_threshold,
                'per_chunk': all_stats,
                'total_kept': int(len(gx_he)),
            }

            if self.verbose:
                print(f'\n  total kept (all chunks): {len(gx_he):,}')

        # ================================================================
        # 공통: zarr 저장
        # ================================================================
        vf_norm_all = np.asarray(vf_norm_all)
        self.dataset.downsampled_vectors = (gx_he, gy_he, gz_he)
        self.dataset.normalized_vectors = normalized_vectors
        self.dataset.downsampled_vf_norm = vf_norm_all
        self.dataset.coordinate_offset = coordinate_offset
        self.dataset.sampling_distance = sampling_distance
        self.dataset.genes = genes

        for key in ('downsampled_vectors', 'normalized_vectors',
                    'downsampled_vf_norm', 'genes'):
            if key in self.dataset.zarr_group:
                del self.dataset.zarr_group[key]

        self.dataset.zarr_group['downsampled_vectors'] = np.stack([gx_he, gy_he, gz_he], axis=0)
        self.dataset.zarr_group['normalized_vectors'] = normalized_vectors
        # per-vector local KDE density, row-aligned with normalized_vectors
        self.dataset.zarr_group['downsampled_vf_norm'] = vf_norm_all
        self.dataset.zarr_group['genes'] = np.array(genes)

        # Persist the (corrected) coordinate metadata. In chunked mode the merge
        # above fixed coordinate_offset / bounds to global values; re-write them
        # so the store no longer holds the last chunk's single-tile metadata.
        for key in ('coordinate_offset', 'coordinate_bounds',
                    'original_coordinate_bounds'):
            value = getattr(self.dataset, key, None)
            if value is None:
                continue
            if key in self.dataset.zarr_group:
                del self.dataset.zarr_group[key]
            self.dataset.zarr_group[key] = np.array(value)

        return sample_stats

    def map_celltypes(self, sdata=None, centroids=None, element_name=None,
                      method='corr', exclude_gene_indices=None,
                      bandwidth=2.5, sampling_distance=1.0, prune_coefficient=4.3,
                      size_after_normalization=10, log_transform=True, overwrite=True,
                      gpu_device='cuda', force_cpu=False,
                      query_subchunk_size=None, gpu_target_bytes=int(1e9),
                      map_kwargs=None, save_vf=False):
        """Single entry point for celltype mapping.

        Two regimes, auto-detected from the store:

        * **count** (blur ``vf_normalized`` present) — the field is read from
          zarr and mapped by ``method``: ``'corr'`` (centroid Pearson) or
          ``'tsvm'`` (linear/transductive SVM). No KDE.
        * **transcript** (no precomputed field) — KDE is computed chunk-by-chunk
          and correlated with centroids, dispatching to
          :meth:`map_celltypes_gpu` (CUDA available and not ``force_cpu``) or
          :meth:`map_celltypes_cpu`. Only ``method='corr'`` is supported here.

        Parameters
        ----------
        method : {'corr', 'tsvm'}, default 'corr'
            Mapping algorithm (count regime only; transcript is always 'corr').
            ``'corr'`` writes the per-voxel score to ``max_correlations``
            (filtered by ``min_r``); ``'tsvm'`` writes the softmax probability to
            ``max_probabilities`` (filtered by ``min_p``). ``'tsvm'`` defaults to
            **PCA-50 + RBF kernel** (random Fourier features) and, unlike ``'corr'``,
            can **abstain**: a low-confidence voxel is left Unknown (-1). Override via
            ``map_kwargs={'fit_kwargs': {...}}`` (e.g. ``{'kernel': 'linear'}``, or
            ``{'n_components': None, 'kernel': 'linear'}`` for full-gene mode).
        map_kwargs : dict, optional
            Extra kwargs forwarded to the count mapper
            (:func:`cellman.spatial._count.map_corr` /
            :func:`cellman.spatial._tsvm.map_tsvm`).
        gpu_device : str, default ``'cuda'``
            CUDA device used by the GPU path.
        force_cpu : bool, default False
            If True, always use the CPU path regardless of GPU availability.
        save_vf : bool, default False
            **Transcript regime only.** By default the imaging-based path
            (Xenium / MERFISH) computes the KDE vector field chunk-by-chunk,
            maps it on the fly, and DISCARDS the field (only ``celltype_maps`` /
            ``max_correlations`` / ``vf_norm`` are persisted). When True, the
            normalized field is also written to ``vf_normalized``
            ``(n_genes_use, vx, vy, vz)`` in zarr (gene axis first, same
            convention as the count backend), plus ``vf_normalized_genes`` naming
            the gene axis. Ignored in the count regime — array-based data
            (Visium HD / Stereo-seq) already stores ``vf_normalized`` at
            ``compute_vf`` time. Note: this holds one extra CPU buffer of
            ``n_genes_use × chunk_x × vy × vz`` floats per chunk and adds the
            field's on-disk size, so enable it only when you need the field.
        query_subchunk_size, gpu_target_bytes
            Forwarded to :meth:`map_celltypes_gpu` (ignored on CPU).
        Other parameters are shared by the transcript paths; see
        :meth:`map_celltypes_cpu`.

        Returns
        -------
        None
            The map is written into the store (``celltype_maps`` +
            ``max_correlations`` in zarr, and ``ds.celltype_maps`` /
            ``ds.max_correlations`` in memory); nothing is returned.
        """
        # ---- count regime: map over the precomputed blur field ----
        zg = self.dataset.zarr_group
        if ('vf_normalized' in zg) and (zg.attrs.get('backend') == 'blur'):
            mk = dict(map_kwargs or {})
            mk.setdefault('device', gpu_device)
            mk.setdefault('overwrite', overwrite)
            if method == 'corr':
                from ._count import map_corr
                map_corr(self, centroids=centroids, **mk)
            elif method == 'tsvm':
                from ._tsvm import map_tsvm
                map_tsvm(self, device=mk.get('device', gpu_device),
                         overwrite=mk.get('overwrite', overwrite),
                         fit_kwargs=mk.get('fit_kwargs'),
                         apply_kwargs=mk.get('apply_kwargs'))
            else:
                raise ValueError(f"unknown mapping method {method!r} "
                                 "(count: 'corr'|'tsvm')")
            # results are persisted to zarr (corr → max_correlations, tsvm →
            # max_probabilities); nothing is returned so the notebook cell
            # doesn't dump the array.
            return None

        if method != 'corr':
            raise ValueError(f"method={method!r} needs a precomputed blur field "
                             "(count data); transcript mapping is 'corr' only.")
        if sdata is None:
            raise ValueError("transcript map_celltypes requires sdata.")
        shared = dict(
            centroids=centroids, element_name=element_name,
            exclude_gene_indices=exclude_gene_indices, bandwidth=bandwidth,
            sampling_distance=sampling_distance, prune_coefficient=prune_coefficient,
            size_after_normalization=size_after_normalization,
            log_transform=log_transform, overwrite=overwrite, save_vf=save_vf,
        )
        use_gpu = False
        if not force_cpu:
            try:
                from ._kde_gpu import gpu_available
                use_gpu = gpu_available(gpu_device)
            except Exception:
                use_gpu = False
        if use_gpu:
            self._m(f'map_celltypes: GPU path ({gpu_device})')
            self._map_celltypes_gpu(
                sdata, gpu_device=gpu_device,
                query_subchunk_size=query_subchunk_size,
                gpu_target_bytes=gpu_target_bytes, **shared,
            )
        else:
            self._m('map_celltypes: CPU path')
            self._map_celltypes_cpu(sdata, **shared)
        return None

    def _map_celltypes_cpu(self, sdata, centroids=None, element_name=None,
                         exclude_gene_indices=None,
                         bandwidth=2.5, sampling_distance=1.0, prune_coefficient=4.3,
                         size_after_normalization=10, log_transform=True, overwrite=True,
                         save_vf=False):
        """
        Stream-compute celltype_maps for H&E-sampled data.

        Reuses ``self.dataset.shape`` set by ``downsample_vector`` (or
        ``_prepare_grid``) so that SpatialData is NOT re-parsed for grid
        dimensions.  KDE is computed chunk-by-chunk directly from
        ``self.locations`` (already stored by ``_prepare_grid``).

        Parameters
        ----------
        sdata : spatialdata.SpatialData
            Only used to populate ``self.locations`` if not already set.
        centroids : ndarray (n_clusters, n_genes), optional
            Falls back to ``self.dataset.centroids``.

        Returns
        -------
        celltype_maps, max_correlations : zarr arrays
        """
        from tqdm.auto import tqdm

        # Load latest state from zarr store
        self.dataset.cluster_labels          = self.dataset.zarr_group['cluster_labels'][:]
        self.dataset.downsampled_vectors              = self.dataset.zarr_group['downsampled_vectors'][:]
        self.dataset.normalized_vectors      = self.dataset.zarr_group['normalized_vectors'][:]
        self.dataset.filtered_cluster_labels = self.dataset.zarr_group['filtered_cluster_labels'][:]
        self.dataset.centroids               = self.dataset.zarr_group['centroids'][:]
        self.dataset.centroids_stdev         = self.dataset.zarr_group['centroids_stdev'][:]

        # ---- centroids ----
        if centroids is None:
            if getattr(self.dataset, 'centroids', None) is None:
                raise ValueError('centroids not found. Run cluster_vectors() first.')
            centroids = np.asarray(self.dataset.centroids, dtype=np.float32)
        else:
            centroids = np.asarray(centroids, dtype=np.float32)

        # ---- grid shape: reuse if already computed ----
        if getattr(self.dataset, 'shape', None) is None:
            self._prepare_grid(
                sdata,
                element_name=element_name,
                bandwidth=bandwidth,
                sampling_distance=sampling_distance,
                prune_coefficient=prune_coefficient,
                size_after_normalization=size_after_normalization,
            )
        else:
            # locations must be set; if missing (e.g. fresh load), re-parse
            if not hasattr(self, 'locations') or self.locations is None:
                self._prepare_grid(
                    sdata,
                    element_name=element_name,
                    bandwidth=bandwidth,
                    sampling_distance=sampling_distance,
                    prune_coefficient=prune_coefficient,
                    size_after_normalization=size_after_normalization,
                )

        vx, vy, vz = (int(self.dataset.shape[0]),
                      int(self.dataset.shape[1]),
                      int(self.dataset.shape[2]))

        self.dataset.bandwidth = bandwidth
        self.dataset.ndim = 2 if vz == 1 else 3
        self.dataset.expression_threshold = (
            1 / (np.sqrt(2 * np.pi) * bandwidth) ** self.dataset.ndim
        )
        self.dataset.norm_threshold = self.dataset.expression_threshold * 2

        # ---- resolve gene list ----
        grouped = self.locations.groupby('gene', sort=True)
        grouped_map = {gene: loc for gene, loc in grouped}

        if getattr(self.dataset, 'genes', None) is None:
            genes = sorted(grouped_map.keys())
            self.dataset.genes = genes
        else:
            genes = list(self.dataset.genes)

        if centroids.shape[1] != len(genes):
            raise ValueError(
                f'centroids gene dim ({centroids.shape[1]}) != gene count ({len(genes)})'
            )

        shared_gene_indices = [i for i, g in enumerate(genes) if g in grouped_map]
        shared_genes = [genes[i] for i in shared_gene_indices]
        missing_genes = [g for g in genes if g not in grouped_map]
        if len(shared_genes) == 0:
            raise ValueError('No common genes between input data and centroids.')
        if missing_genes and self.verbose:
            print(f'map_celltypes_cpu: dropping {len(missing_genes)} missing genes '
                  f'(e.g. {missing_genes[:5]})')

        grouped_list = [(g, grouped_map[g]) for g in shared_genes]
        centroids_shared = centroids[:, shared_gene_indices]

        if exclude_gene_indices is not None:
            idx_to_local = {gi: li for li, gi in enumerate(shared_gene_indices)}
            exclude_shared = sorted(
                idx_to_local[i] for i in exclude_gene_indices if i in idx_to_local
            )
            centroids_use = np.delete(centroids_shared, exclude_shared, axis=1)
        else:
            exclude_shared = None
            centroids_use = centroids_shared

        # ---- zarr setup ----
        def _safe_remove(name):
            if name in self.dataset.zarr_group:
                del self.dataset.zarr_group[name]

        if overwrite:
            for name in ['vf_norm', 'celltype_maps', 'max_correlations',
                         'vf_normalized', 'vf_normalized_genes']:
                _safe_remove(name)

        import numcodecs as _numcodecs
        _safe_remove('genes')
        self.dataset.zarr_group['genes'] = np.asarray(genes)

        self.dataset._zarr_create_array(self.dataset.zarr_group, 'vf_norm', (vx, vy, vz), 'f4')
        self.dataset._zarr_create_array(self.dataset.zarr_group, 'celltype_maps', (vx, vy, vz), 'i4')
        self.dataset._zarr_create_array(self.dataset.zarr_group, 'max_correlations', (vx, vy, vz), 'f4')

        # gene axis of the mapped field = shared genes minus the excluded ones
        # (matches ``centroids_use`` / the GPU path's ``use_genes``).
        use_genes = ([shared_genes[i] for i in range(len(shared_genes))
                      if i not in set(exclude_shared)]
                     if exclude_shared is not None else list(shared_genes))
        n_genes_use = int(centroids_use.shape[1])
        _cpu_vf_ts = None; _cpu_vf_zarr = None; _cpu_inflight = []
        if save_vf:
            # opt-in: persist the on-the-fly KDE field, gene axis first (like the
            # count backend). Prefer tensorstore's C++ async writer (bit-identical,
            # same zarr-v3 metadata); fall back to zarr-python.
            from ._count import _tensorstore_array
            _safe_remove('vf_normalized'); _safe_remove('vf_normalized_genes')
            self.dataset.zarr_group['vf_normalized_genes'] = np.asarray(use_genes)
            _sr = getattr(self.dataset.zarr_group.store, 'root', None)
            _gd = (os.path.join(str(_sr), self.dataset.zarr_group.path)
                   if self.dataset.zarr_group.path else str(_sr)) if _sr is not None else None
            _shape = (n_genes_use, vx, vy, vz); _chunks = (n_genes_use, 1, vy, vz)
            _cpu_vf_ts = _tensorstore_array(_gd, 'vf_normalized', _shape, _chunks)
            if _cpu_vf_ts is None:
                _cpu_vf_zarr = self.dataset.zarr_group.create_array(
                    'vf_normalized', shape=_shape, chunks=_chunks, dtype='f4', fill_value=0.0)

        cx, n_chunk = self.calc_chunk(vx, vy, vz, len(shared_genes), mode='map_celltypes_cpu')
        if cx is None or cx <= 0:
            raise RuntimeError('Cannot determine chunk size.')

        import numcodecs as _nc2
        _nc2.blosc.set_nthreads(self.ncores)

        # ---- chunk loop ----
        for chunk_idx in range(n_chunk):
            x0 = chunk_idx * cx
            x1 = min(vx, x0 + cx)
            local_cx = x1 - x0

            vf_chunk = np.zeros((local_cx, vy, vz, len(shared_genes)), dtype='f4')

            if self.verbose:
                gene_bar = tqdm(
                    total=len(grouped_list),
                    desc=f'KDE genes {chunk_idx + 1}/{n_chunk}',
                    leave=True, dynamic_ncols=True,
                )
            else:
                gene_bar = None

            for gidx, (_, loc) in enumerate(grouped_list):
                locs = np.asarray(loc)
                if locs.shape[1] == 2:
                    loc_x, loc_y = locs[:, 0].astype(np.float64), locs[:, 1].astype(np.float64)
                    loc_z = np.zeros(locs.shape[0], dtype=np.float64)
                    dim = 2
                else:
                    loc_x, loc_y, loc_z = (locs[:, 0].astype(np.float64),
                                           locs[:, 1].astype(np.float64),
                                           locs[:, 2].astype(np.float64))
                    dim = 3

                accumulate_gene_vector(
                    vf_chunk,
                    loc_x=loc_x, loc_y=loc_y, loc_z=loc_z,
                    sampling_distance=float(sampling_distance),
                    bandwidth=float(bandwidth),
                    prune_coefficient=float(prune_coefficient),
                    gidx=gidx, dim=int(dim), ncores=int(self.ncores),
                    x0=int(x0), y0=0, z0=0,
                )
                if gene_bar is not None:
                    gene_bar.update(1)

            if gene_bar is not None:
                gene_bar.close()

            vf_norm_chunk = vf_chunk.sum(axis=3)
            self.dataset.zarr_group['vf_norm'][x0:x1, :, :] = vf_norm_chunk

            if save_vf:
                # normalize the FULL field (all voxels, all shared genes), drop the
                # excluded gene columns, then store gene-first. Independent of the
                # kept-voxel mapping below → celltype map unaffected by save_vf.
                vf_full = (vf_chunk
                           / np.clip(vf_norm_chunk[..., None], 1e-12, None)
                           * float(size_after_normalization))
                if log_transform:
                    np.log1p(vf_full, out=vf_full)
                if exclude_shared is not None:
                    vf_full = np.delete(vf_full, exclude_shared, axis=3)
                vf_full = np.ascontiguousarray(np.moveaxis(vf_full, 3, 0))
                if _cpu_vf_ts is not None:
                    _cpu_inflight.append(_cpu_vf_ts[:, x0:x1, :, :].write(vf_full))
                    while len(_cpu_inflight) > 4:
                        _cpu_inflight.pop(0).result()
                else:
                    _cpu_vf_zarr[:, x0:x1, :, :] = vf_full
                del vf_full

            local_ctmap = np.full((local_cx, vy, vz), -1, dtype=np.int32)
            local_corr = np.full((local_cx, vy, vz), -1, dtype=np.float32)

            x_idx, y_idx, z_idx, data_chunk = normalize_vf(
                vf_chunk, vf_norm_chunk, size_after_normalization, x0, self.ncores
            )
            if log_transform:
                np.log1p(data_chunk, out=data_chunk)

            if data_chunk.shape[0] > 0:
                data_use = (np.delete(data_chunk, exclude_shared, axis=1)
                            if exclude_shared is not None else data_chunk)
                best_corr = np.full(data_use.shape[0], -1, dtype=np.float32)
                best_idx = np.full(data_use.shape[0], -1, dtype=np.int32)

                if self.verbose:
                    corr_bar = tqdm(
                        total=len(centroids_use),
                        desc=f'Correlations {chunk_idx + 1}/{n_chunk}',
                        leave=True, dynamic_ncols=True,
                    )
                else:
                    corr_bar = None

                for cidx, centroid in enumerate(centroids_use):
                    corr_chunk = calc_ctmap(centroid, data_use, self.ncores)
                    corr_chunk = np.nan_to_num(corr_chunk).astype(np.float32, copy=False)
                    improved = corr_chunk > best_corr
                    best_corr[improved] = corr_chunk[improved]
                    best_idx[improved] = cidx
                    if corr_bar is not None:
                        corr_bar.update(1)

                if corr_bar is not None:
                    corr_bar.close()

                local_x = x_idx.astype(np.int64) - x0
                local_y = y_idx.astype(np.int64)
                local_z = z_idx.astype(np.int64)
                local_ctmap[local_x, local_y, local_z] = best_idx
                local_corr[local_x, local_y, local_z] = best_corr

            self.dataset.zarr_group['celltype_maps'][x0:x1, :, :] = local_ctmap
            self.dataset.zarr_group['max_correlations'][x0:x1, :, :] = local_corr

            del vf_chunk, vf_norm_chunk, x_idx, y_idx, z_idx, data_chunk
            del local_ctmap, local_corr
            gc.collect()

        for _f in _cpu_inflight:                        # flush pending tensorstore writes
            _f.result()
        import dask.array as _da
        self.dataset._vf_norm = _da.from_zarr(self.dataset.zarr_group['vf_norm'])
        self.dataset.celltype_maps = self.dataset.zarr_group['celltype_maps'][:]
        self.dataset.max_correlations = self.dataset.zarr_group['max_correlations'][:]

    def filter_celltypemaps(self, min_p=0.6, min_r=0.6, min_norm=0.1, fill_blobs=True, min_blob_area=0, filter_params={}, output_mask=None):
        """
        Post-filter cell type maps created by `map_celltypes`.

        :param min_r: minimum threshold of the correlation.
        :type min_r: float
        :param min_norm: minimum threshold of the vector norm.
            If a string is given instead, then the threshold is automatically determined using
            sklearn's `threshold filter functions <https://scikit-image.org/docs/dev/api/skimage.filters.html>`_ (The functions start with `threshold_`).
        :type min_norm: str or float
        :param min_blob_area: The blobs with its area less than this value will be removed.
        :type min_blob_area: int
        :param filter_params: Filter parameters used for the sklearn's threshold filter functions.
            Not used when `min_norm` is float.
        :type filter_params: dict
        :param output_mask: If given, the cell type maps will be filtered using the output mask.
        :type output_mask: np.ndarray(bool)
        """

        # Pick the per-voxel score volume (probability or correlation) and its
        # threshold once. ``get_celltype_*`` builds a full-volume array per
        # cell type, so we read the underlying volume directly instead.
        if self.dataset.max_probabilities is not None:
            score_vol = self.dataset.max_probabilities
            if score_vol.ndim == 4:
                score_vol = score_vol[..., 0]
            r_thr = min_p
        else:
            score_vol = self.dataset.max_correlations
            r_thr = min_r

        celltype_maps = self.dataset.celltype_maps

        if isinstance(min_norm, str):
            # Adaptive per-slice threshold path (rare). Kept close to the
            # original, but the score is read from ``score_vol`` masked by the
            # cell-type id rather than rebuilding a full volume per cell type.
            _filter_params = filter_params.copy()
            # _filter_params dict will be used for kwd params for filter_* functions.
            # some functions doesn't support param 'offset', therefore remove it from here
            filter_offset = _filter_params.pop('offset', 0)

            mask = np.zeros(self.dataset.vf_norm.shape, dtype=bool)
            for cidx in np.unique(celltype_maps):
                if cidx == -1:
                    continue
                ctcorr = np.full_like(score_vol, -1.0)
                cidx_sel = celltype_maps == cidx
                ctcorr[cidx_sel] = score_vol[cidx_sel]
                for z in range(self.dataset.shape[2]):
                    vf_norm_z = self.dataset.vf_norm[..., z].compute()
                    ctcorr_mask_z = ctcorr[..., z] > r_thr
                    if min_norm in ["local", "niblack", "sauvola", "localotsu"]:
                        im = np.zeros(vf_norm_z.shape)
                        im[ctcorr_mask_z] = vf_norm_z[ctcorr_mask_z]
                    if min_norm == "localotsu":
                        max_norm = np.max(im)
                        im /= max_norm
                        selem = disk(_filter_params['radius'])
                        min_norm_cut = filters.rank.otsu(im, selem) * max_norm
                    else:
                        filter_func = getattr(filters, "threshold_" + min_norm)
                        if min_norm in ["local", "niblack", "sauvola"]:
                            min_norm_cut = filter_func(im, **_filter_params)
                        else:
                            highr_norm = vf_norm_z[ctcorr_mask_z]
                            #sigma = np.std(highr_norm)
                            if len(highr_norm) == 0 or np.max(highr_norm) == np.min(highr_norm):
                                min_norm_cut = np.max(self.dataset.vf_norm)
                            else:
                                min_norm_cut = filter_func(highr_norm, **_filter_params)
                    min_norm_cut += filter_offset # manually apply filter offset
                    mask[..., z][np.logical_and(vf_norm_z > min_norm_cut, ctcorr_mask_z)] = 1
        else:
            # Fast vectorized path for a numeric ``min_norm``. The original
            # per-cell-type union reduces exactly to this single expression:
            # a voxel is kept iff it has a (non-background) cell type whose
            # score and vector-field norm both pass their thresholds.
            vf_norm = np.asarray(self.dataset.vf_norm)
            # vf_norm is stored 2D as (n_R, n_C) while score_vol / celltype_maps
            # are (n_C, n_R, 1). Without aligning orientation+ndim, the elementwise
            # `&` below broadcasts to (n_C, n_R, n_C) -- O(n_C^2 * n_R), ~125GB on
            # large maps (minutes + swap, wrong-shaped mask). Align to match.
            if vf_norm.ndim == 2 and score_vol.ndim == 3:
                vf_norm = vf_norm.T[..., None]
            mask = (vf_norm > min_norm) & (score_vol > r_thr) & (celltype_maps != -1)

        # Blob post-processing — run ONCE on the final mask instead of once per
        # cell type on the growing cumulative mask (the original O(N_celltypes)
        # bottleneck).
        if min_blob_area > 0:
            blob_labels = measure.label(mask, background=0)
            for bp in measure.regionprops(blob_labels):
                if bp.area_filled < min_blob_area:
                    mask[tuple(bp.coords.T)] = 0  # remove small blob
                    continue
                if fill_blobs and bp.area != bp.area_filled:
                    minx, miny, minz, maxx, maxy, maxz = bp.bbox
                    mask[minx:maxx, miny:maxy, minz:maxz] |= bp.image_filled
        elif fill_blobs:
            # Fill enclosed holes in one optimized pass (GPU when available;
            # bit-identical boolean morphology, auto CPU fallback).
            mask = _binary_fill_holes_fast(mask)

        filtered_ctmaps = np.array(celltype_maps, copy=True)
        filtered_ctmaps[mask == False] = -1

        if output_mask is not None:
            filtered_ctmaps[~output_mask.astype(bool)] = -1
        self.dataset.filtered_celltype_maps = filtered_ctmaps
        self.dataset.zarr_group['filtered_celltype_maps'] = self.dataset.filtered_celltype_maps

    # ------------------------------------------------------------------
    # GPU equivalent of map_celltypes_cpu
    # ------------------------------------------------------------------

    def _map_celltypes_gpu(self, sdata, centroids=None, element_name=None,
                          exclude_gene_indices=None,
                          bandwidth=2.5, sampling_distance=1.0,
                          prune_coefficient=4.3, size_after_normalization=10,
                          log_transform=True, overwrite=True,
                          gpu_device='cuda', query_subchunk_size=None,
                          gpu_target_bytes=int(1e9), save_vf=False):
        """GPU equivalent of :meth:`map_celltypes_cpu` (single machine).

        Reads zarr state, KDE-computes the vector field chunk-by-chunk on
        a CUDA device (Triton kernel + torch GEMM Pearson correlation),
        and writes ``vf_norm``, ``celltype_maps``, ``max_correlations``.

        Numerical equivalence vs the CPU path: not bit-identical; expected
        per-voxel correlation diff <1e-5, with minor label flips possible
        at decision boundaries (much less than 0.1% of voxels for
        well-separated centroids).

        Parameters
        ----------
        gpu_device : str, default ``'cuda'``
        query_subchunk_size : int, optional
            Voxel queries per Triton launch; auto-sized from
            ``gpu_target_bytes / (n_genes_use * 4)`` when None.
        gpu_target_bytes : int, default 1 GB
            Target VRAM budget for the per-sub-chunk output buffer.
        Other parameters are identical to :meth:`map_celltypes_cpu`.

        Now backed by the producer/consumer split: KDE field production
        (:meth:`_kde_field_iter_gpu`) is separated from the shared Pearson-max
        mapper (``_pearson_max_gpu``, identical to the count path), with the full
        normalized field optionally teed to a tensorstore writer (``save_vf``) in
        the same streaming pass. Numerically equivalent to the legacy fused core
        (vf_norm bit-identical; correlation diff <1e-5 from matmul batching).
        """
        return self._transcript_gpu(
            sdata, do_map=True, save_vf=save_vf,
            centroids=centroids, element_name=element_name,
            exclude_gene_indices=exclude_gene_indices, bandwidth=bandwidth,
            sampling_distance=sampling_distance, prune_coefficient=prune_coefficient,
            size_after_normalization=size_after_normalization,
            log_transform=log_transform, overwrite=overwrite,
            gpu_device=gpu_device, query_subchunk_size=query_subchunk_size,
            gpu_target_bytes=gpu_target_bytes,
        )

    # Backward-compat aliases: the CPU/GPU implementations are now internal
    # branches of :meth:`map_celltypes` (the single public entry point). These
    # public names are kept so older callers (e.g. xenium notebooks that invoke
    # ``analysis.map_celltypes_cpu(...)`` directly) keep working; prefer
    # ``map_celltypes(..., force_cpu=...)`` in new code.
    map_celltypes_cpu = _map_celltypes_cpu
    map_celltypes_gpu = _map_celltypes_gpu

    def _map_celltypes_gpu_core(self, sdata, *, chunk_start=None, chunk_end=None,
                                cx=None, create_arrays=True, attach=True,
                                centroids=None, element_name=None,
                                exclude_gene_indices=None,
                                bandwidth=2.5, sampling_distance=1.0,
                                prune_coefficient=4.3, size_after_normalization=10,
                                log_transform=True, overwrite=True,
                                gpu_device='cuda', query_subchunk_size=None,
                                gpu_target_bytes=int(1e9), save_vf=False):
        """Shared GPU mapping core over a chunk range ``[chunk_start, chunk_end)``.

        Backs both :meth:`map_celltypes_gpu` (full run, single machine:
        ``create_arrays=True, attach=True``) and the GPU branch of
        :meth:`map_celltypes_partial` (per-worker range:
        ``create_arrays=False, attach=False``, ``cx`` supplied by
        :meth:`init_map_celltypes`). When ``cx`` is None it is computed
        via :meth:`calc_chunk`; otherwise the supplied value is used so
        all workers agree on the chunk grid.
        """
        from tqdm.auto import tqdm
        try:
            import torch
            from ._mapping_gpu import (
                _kde_setup_celllist, _kde_run_subchunk, _pearson_corr_gpu,
            )
        except ImportError as e:
            raise ImportError(
                "map_celltypes_gpu requires torch + triton + "
                "._mapping_gpu."
            ) from e
        if not torch.cuda.is_available():
            raise RuntimeError("map_celltypes_gpu requires a CUDA device")

        # ---- load zarr state (same as map_celltypes_cpu) ----
        for key in ('cluster_labels', 'downsampled_vectors', 'normalized_vectors',
                    'filtered_cluster_labels', 'centroids', 'centroids_stdev'):
            if key in self.dataset.zarr_group:
                setattr(self.dataset, key,
                        self.dataset.zarr_group[key][:])

        # ---- centroids ----
        if centroids is None:
            if getattr(self.dataset, 'centroids', None) is None:
                raise ValueError('centroids not found. Run cluster_vectors() first.')
            centroids = np.asarray(self.dataset.centroids, dtype=np.float32)
        else:
            centroids = np.asarray(centroids, dtype=np.float32)

        # ---- grid shape ----
        if getattr(self.dataset, 'shape', None) is None or \
                not hasattr(self, 'locations') or self.locations is None:
            self._prepare_grid(
                sdata, element_name=element_name,
                bandwidth=bandwidth, sampling_distance=sampling_distance,
                prune_coefficient=prune_coefficient,
                size_after_normalization=size_after_normalization,
            )

        vx, vy, vz = (int(self.dataset.shape[0]),
                      int(self.dataset.shape[1]),
                      int(self.dataset.shape[2]))

        self.dataset.bandwidth = bandwidth
        self.dataset.ndim = 2 if vz == 1 else 3
        self.dataset.expression_threshold = (
            1 / (np.sqrt(2 * np.pi) * bandwidth) ** self.dataset.ndim
        )
        self.dataset.norm_threshold = self.dataset.expression_threshold * 2
        norm_threshold = float(self.dataset.norm_threshold)

        # ---- gene list / shared / exclude ----
        grouped = self.locations.groupby('gene', sort=True)
        grouped_map = {gene: loc for gene, loc in grouped}

        if getattr(self.dataset, 'genes', None) is None:
            genes = sorted(grouped_map.keys())
            self.dataset.genes = genes
        else:
            genes = list(self.dataset.genes)

        if centroids.shape[1] != len(genes):
            raise ValueError(
                f'centroids gene dim ({centroids.shape[1]}) != gene count ({len(genes)})'
            )

        shared_gene_indices = [i for i, g in enumerate(genes) if g in grouped_map]
        shared_genes = [genes[i] for i in shared_gene_indices]
        missing_genes = [g for g in genes if g not in grouped_map]
        if len(shared_genes) == 0:
            raise ValueError('No common genes between input data and centroids.')
        if missing_genes and self.verbose:
            print(f'map_celltypes_gpu: dropping {len(missing_genes)} missing genes '
                  f'(e.g. {missing_genes[:5]})')

        centroids_shared = centroids[:, shared_gene_indices]

        if exclude_gene_indices is not None:
            idx_to_local = {gi: li for li, gi in enumerate(shared_gene_indices)}
            exclude_shared = sorted(
                idx_to_local[i] for i in exclude_gene_indices if i in idx_to_local
            )
        else:
            exclude_shared = None

        if exclude_shared:
            keep_in_shared = [i for i in range(len(shared_genes))
                              if i not in set(exclude_shared)]
            use_genes = [shared_genes[i] for i in keep_in_shared]
            centroids_use = centroids_shared[:, keep_in_shared]
        else:
            use_genes = list(shared_genes)
            centroids_use = centroids_shared
        n_genes_use = centroids_use.shape[1]

        # ---- zarr setup (same as map_celltypes_cpu) ----
        def _safe_remove(name):
            if name in self.dataset.zarr_group:
                del self.dataset.zarr_group[name]

        if create_arrays:
            if overwrite:
                for name in ('vf_norm', 'celltype_maps', 'max_correlations',
                             'vf_normalized', 'vf_normalized_genes'):
                    _safe_remove(name)
            import numcodecs as _numcodecs
            _safe_remove('genes')
            self.dataset.zarr_group['genes'] = np.asarray(genes)
            self.dataset._zarr_create_array(self.dataset.zarr_group, 'vf_norm', (vx, vy, vz), 'f4')
            self.dataset._zarr_create_array(self.dataset.zarr_group, 'celltype_maps', (vx, vy, vz), 'i4')
            self.dataset._zarr_create_array(self.dataset.zarr_group, 'max_correlations', (vx, vy, vz), 'f4')
            if save_vf:
                # opt-in: persist the on-the-fly KDE field (gene axis first, like
                # the count backend's vf_normalized). x is chunked 1-plane-wide so
                # each per-chunk write [:, x0:x1] spans whole store chunks (no RMW).
                _safe_remove('vf_normalized'); _safe_remove('vf_normalized_genes')
                self.dataset.zarr_group.create_array(
                    'vf_normalized', shape=(n_genes_use, vx, vy, vz),
                    chunks=(n_genes_use, 1, vy, vz), dtype='f4', fill_value=0.0)
                self.dataset.zarr_group['vf_normalized_genes'] = np.asarray(use_genes)
        elif save_vf and 'vf_normalized' not in self.dataset.zarr_group:
            raise ValueError("save_vf=True with create_arrays=False requires "
                             "'vf_normalized' to be pre-created (distributed path).")

        cx_calc, _ = self.calc_chunk(vx, vy, vz, n_genes_use, mode='map_celltypes_cpu')
        if cx is None:
            cx = cx_calc
        if cx is None or cx <= 0:
            raise RuntimeError('Cannot determine chunk size.')
        n_chunk = (vx + cx - 1) // cx
        if chunk_start is None:
            chunk_start = 0
        if chunk_end is None:
            chunk_end = n_chunk

        # ---- auto-size sub-chunk to fit GPU output buffer ----
        if query_subchunk_size is None:
            query_subchunk_size = max(1024, int(gpu_target_bytes // (n_genes_use * 4)))

        # ---- Build flat transcript arrays for use_genes ----
        all_loc_x, all_loc_y, all_loc_z, all_gi = [], [], [], []
        for gidx_use, g in enumerate(use_genes):
            locs = np.asarray(grouped_map[g], dtype=np.float64)
            n_t = locs.shape[0]
            if n_t == 0:
                continue
            all_loc_x.append(locs[:, 0])
            all_loc_y.append(locs[:, 1])
            if locs.shape[1] >= 3:
                all_loc_z.append(locs[:, 2])
            else:
                all_loc_z.append(np.zeros(n_t, dtype=np.float64))
            all_gi.append(np.full(n_t, gidx_use, dtype=np.int64))
        if all_loc_x:
            loc_x_np = np.concatenate(all_loc_x)
            loc_y_np = np.concatenate(all_loc_y)
            loc_z_np = np.concatenate(all_loc_z)
            gi_np    = np.concatenate(all_gi)
        else:
            loc_x_np = np.zeros(0, dtype=np.float64)
            loc_y_np = np.zeros(0, dtype=np.float64)
            loc_z_np = np.zeros(0, dtype=np.float64)
            gi_np    = np.zeros(0, dtype=np.int64)
        del all_loc_x, all_loc_y, all_loc_z, all_gi

        # ---- Move transcripts + centroids to GPU once ----
        loc_x_t = torch.as_tensor(loc_x_np, dtype=torch.float64, device=gpu_device)
        loc_y_t = torch.as_tensor(loc_y_np, dtype=torch.float64, device=gpu_device)
        loc_z_t = torch.as_tensor(loc_z_np, dtype=torch.float64, device=gpu_device)
        gi_t    = torch.as_tensor(gi_np,    dtype=torch.int64,   device=gpu_device)
        centroids_t = torch.as_tensor(centroids_use, dtype=torch.float32, device=gpu_device)

        sd_um = float(sampling_distance)
        dim = 2 if vz == 1 else 3

        import numcodecs as _nc2
        _nc2.blosc.set_nthreads(self.ncores)

        # ---- chunk loop ----
        for chunk_idx in range(chunk_start, chunk_end):
            x0 = chunk_idx * cx
            x1 = min(vx, x0 + cx)
            local_cx = x1 - x0

            # Voxel center grid for this chunk (in same µm as transcripts)
            qx_grid = (np.arange(x0, x1) * sd_um).astype(np.float64)
            qy_grid = (np.arange(vy) * sd_um).astype(np.float64)
            qz_grid = (np.arange(vz) * sd_um).astype(np.float64)
            Qx, Qy, Qz = np.meshgrid(qx_grid, qy_grid, qz_grid, indexing='ij')
            qx_flat = Qx.ravel()
            qy_flat = Qy.ravel()
            qz_flat = Qz.ravel()
            N_total = int(qx_flat.size)

            # CPU-side accumulators for zarr write
            vf_norm_chunk = np.zeros((local_cx, vy, vz), dtype=np.float32)
            ctmap_chunk   = np.full((local_cx, vy, vz), -1, dtype=np.int32)
            corr_chunk    = np.full((local_cx, vy, vz), -1, dtype=np.float32)
            # opt-in full normalized field buffer (gene axis first)
            vf_full_chunk = (np.zeros((n_genes_use, local_cx, vy, vz), dtype=np.float32)
                             if save_vf else None)

            # Build cell list once per chunk (covers chunk's query bbox + transcripts)
            chunk_q_xmin = float(qx_grid.min()) if N_total > 0 else 0.0
            chunk_q_xmax = float(qx_grid.max()) if N_total > 0 else 0.0
            chunk_q_ymin = float(qy_grid.min()) if N_total > 0 else 0.0
            chunk_q_ymax = float(qy_grid.max()) if N_total > 0 else 0.0
            chunk_q_zmin = float(qz_grid.min()) if N_total > 0 else 0.0
            chunk_q_zmax = float(qz_grid.max()) if N_total > 0 else 0.0

            # Filter transcripts to only those within Rw padding of this
            # chunk's x-range. Without this the cell-list spans the full
            # slide x-extent, which makes ``W*H*D`` arange explode (OOM).
            _Rw_pad = float(bandwidth * prune_coefficient + sd_um)
            x_lo_chunk = chunk_q_xmin - _Rw_pad
            x_hi_chunk = chunk_q_xmax + _Rw_pad
            mask_chunk_t = (loc_x_t >= x_lo_chunk) & (loc_x_t <= x_hi_chunk)
            loc_x_chunk_t = loc_x_t[mask_chunk_t]
            loc_y_chunk_t = loc_y_t[mask_chunk_t]
            loc_z_chunk_t = loc_z_t[mask_chunk_t]
            gi_chunk_t    = gi_t[mask_chunk_t]
            del mask_chunk_t

            state = _kde_setup_celllist(
                loc_x_chunk_t, loc_y_chunk_t, loc_z_chunk_t, gi_chunk_t,
                chunk_q_xmin, chunk_q_ymin, chunk_q_zmin,
                chunk_q_xmax, chunk_q_ymax, chunk_q_zmax,
                sd_um, bandwidth, prune_coefficient, dim,
            )

            # Sub-chunk loop
            n_sub = (N_total + query_subchunk_size - 1) // query_subchunk_size
            sub_iter = range(n_sub)
            if self.verbose:
                sub_iter = tqdm(sub_iter,
                                desc=f'KDE+corr chunk {chunk_idx + 1}/{n_chunk}',
                                leave=True, dynamic_ncols=True)
            for s in sub_iter:
                s0 = s * query_subchunk_size
                s1 = min(N_total, s0 + query_subchunk_size)
                sub_qx = qx_flat[s0:s1]
                sub_qy = qy_flat[s0:s1]
                sub_qz = qz_flat[s0:s1]

                kde_t = _kde_run_subchunk(state, sub_qx, sub_qy, sub_qz,
                                          n_genes_use, gpu_device)
                vf_norm_t = kde_t.sum(dim=1)

                # Decompose flat sub indices → (local_x, y, z)
                sub_idx = np.arange(s0, s1)
                local_x = sub_idx // (vy * vz)
                rem = sub_idx % (vy * vz)
                local_y = rem // vz
                local_z = rem % vz

                # Always write vf_norm
                vf_norm_chunk[local_x, local_y, local_z] = (
                    vf_norm_t.detach().cpu().numpy().astype(np.float32, copy=False)
                )

                # opt-in: normalize the FULL field (all voxels, all use_genes) and
                # stash it; independent of the kept-voxel mapping path below so the
                # celltype map is byte-identical whether or not save_vf is set.
                if save_vf:
                    vf_full_t = (kde_t
                                 / torch.clamp(vf_norm_t, min=1e-12).unsqueeze(1)
                                 * float(size_after_normalization))
                    if log_transform:
                        vf_full_t = torch.log1p(vf_full_t)
                    vf_full_chunk[:, local_x, local_y, local_z] = (
                        vf_full_t.T.detach().cpu().numpy().astype(np.float32, copy=False)
                    )
                    del vf_full_t

                mask_t = vf_norm_t > norm_threshold
                n_kept = int(mask_t.sum().item())
                if n_kept == 0:
                    del kde_t, vf_norm_t, mask_t
                    continue

                data_kept_t = kde_t[mask_t]
                vfn_kept_t  = vf_norm_t[mask_t]
                data_kept_t = data_kept_t / vfn_kept_t.unsqueeze(1) * float(size_after_normalization)
                if log_transform:
                    data_kept_t = torch.log1p(data_kept_t)

                corr_t = _pearson_corr_gpu(centroids_t, data_kept_t)  # (k, n_kept)
                best_corr_t, best_idx_t = corr_t.max(dim=0)

                # Mask back to local positions
                mask_np = mask_t.detach().cpu().numpy()
                kept_local_x = local_x[mask_np]
                kept_local_y = local_y[mask_np]
                kept_local_z = local_z[mask_np]
                ctmap_chunk[kept_local_x, kept_local_y, kept_local_z] = (
                    best_idx_t.detach().cpu().numpy().astype(np.int32, copy=False)
                )
                corr_chunk[kept_local_x, kept_local_y, kept_local_z] = (
                    best_corr_t.detach().cpu().numpy().astype(np.float32, copy=False)
                )

                del kde_t, vf_norm_t, mask_t, data_kept_t, vfn_kept_t, corr_t
                del best_corr_t, best_idx_t

            # Write chunk to zarr
            self.dataset.zarr_group['vf_norm'][x0:x1, :, :]          = vf_norm_chunk
            self.dataset.zarr_group['celltype_maps'][x0:x1, :, :]    = ctmap_chunk
            self.dataset.zarr_group['max_correlations'][x0:x1, :, :] = corr_chunk
            if save_vf:
                self.dataset.zarr_group['vf_normalized'][:, x0:x1, :, :] = vf_full_chunk

            del vf_norm_chunk, ctmap_chunk, corr_chunk, state, vf_full_chunk
            del Qx, Qy, Qz, qx_flat, qy_flat, qz_flat
            del loc_x_chunk_t, loc_y_chunk_t, loc_z_chunk_t, gi_chunk_t
            gc.collect()
            torch.cuda.empty_cache()

        # ---- final attach (single-machine convenience) ----
        if attach:
            import dask.array as _da
            self.dataset._vf_norm = _da.from_zarr(self.dataset.zarr_group['vf_norm'])
            self.dataset.celltype_maps = self.dataset.zarr_group['celltype_maps'][:]
            self.dataset.max_correlations = self.dataset.zarr_group['max_correlations'][:]

    # ==================================================================
    # Transcript field: producer / consumer split (GPU)
    #
    #   PRODUCER  `_kde_field_iter_gpu`  — KDE the field chunk-by-chunk,
    #             yielding per x-slab the vf_norm + the KEPT (signal) voxels'
    #             normalized·log1p gene vectors (+ the full field when a caller
    #             wants to persist it). No centroids, no mapping.
    #   CONSUMER  the mapping loop (`_transcript_gpu` below) — Pearson-max via
    #             the SHARED `_pearson_max_gpu` kernel, identical to the count
    #             path's `map_corr`. Optionally tees the full field to a
    #             tensorstore writer (save_vf) in the SAME pass.
    #
    # This is the transcript analogue of compute_vf(count)+map_corr: the field
    # representation (L1-normalized·scale·log1p) and the Pearson-max are the SAME
    # objects the count path uses, so results are independent of how/where the
    # field was produced.
    # ==================================================================
    def _prepare_transcript_field(self, sdata, *, centroids, element_name,
                                  exclude_gene_indices, bandwidth, sampling_distance,
                                  prune_coefficient, size_after_normalization,
                                  gpu_device):
        """Resolve everything the KDE producer/consumer need (grid, gene set,
        centroids, transcripts on GPU). Factored verbatim from the setup block of
        the legacy ``_map_celltypes_gpu_core`` so numerics are preserved."""
        import torch
        for key in ('cluster_labels', 'downsampled_vectors', 'normalized_vectors',
                    'filtered_cluster_labels', 'centroids', 'centroids_stdev'):
            if key in self.dataset.zarr_group:
                setattr(self.dataset, key, self.dataset.zarr_group[key][:])
        # produce-only (compute_vf imaging, no mapping): centroids may be absent —
        # the KDE field is centroid-independent, so use ALL present genes.
        produce_only = centroids is None and getattr(self.dataset, 'centroids', None) is None
        if produce_only:
            centroids = None
        elif centroids is None:
            centroids = np.asarray(self.dataset.centroids, dtype=np.float32)
        else:
            centroids = np.asarray(centroids, dtype=np.float32)

        if getattr(self.dataset, 'shape', None) is None or \
                not hasattr(self, 'locations') or self.locations is None:
            self._prepare_grid(
                sdata, element_name=element_name, bandwidth=bandwidth,
                sampling_distance=sampling_distance, prune_coefficient=prune_coefficient,
                size_after_normalization=size_after_normalization)
        vx, vy, vz = (int(self.dataset.shape[0]), int(self.dataset.shape[1]),
                      int(self.dataset.shape[2]))
        self.dataset.bandwidth = bandwidth
        self.dataset.ndim = 2 if vz == 1 else 3
        self.dataset.expression_threshold = (
            1 / (np.sqrt(2 * np.pi) * bandwidth) ** self.dataset.ndim)
        self.dataset.norm_threshold = self.dataset.expression_threshold * 2
        norm_threshold = float(self.dataset.norm_threshold)

        grouped = self.locations.groupby('gene', sort=True)
        grouped_map = {gene: loc for gene, loc in grouped}
        if getattr(self.dataset, 'genes', None) is None:
            genes = sorted(grouped_map.keys()); self.dataset.genes = genes
        else:
            genes = list(self.dataset.genes)
        if (not produce_only) and centroids.shape[1] != len(genes):
            raise ValueError(
                f'centroids gene dim ({centroids.shape[1]}) != gene count ({len(genes)})')
        shared_gene_indices = [i for i, g in enumerate(genes) if g in grouped_map]
        shared_genes = [genes[i] for i in shared_gene_indices]
        missing_genes = [g for g in genes if g not in grouped_map]
        if len(shared_genes) == 0:
            raise ValueError('No common genes between input data and centroids.')
        if missing_genes and self.verbose:
            print(f'map_celltypes: dropping {len(missing_genes)} missing genes '
                  f'(e.g. {missing_genes[:5]})')
        if produce_only:
            use_genes = list(shared_genes); centroids_use = None
        else:
            centroids_shared = centroids[:, shared_gene_indices]
            if exclude_gene_indices is not None:
                idx_to_local = {gi: li for li, gi in enumerate(shared_gene_indices)}
                exclude_shared = sorted(idx_to_local[i] for i in exclude_gene_indices
                                        if i in idx_to_local)
            else:
                exclude_shared = None
            if exclude_shared:
                keep_in_shared = [i for i in range(len(shared_genes))
                                  if i not in set(exclude_shared)]
                use_genes = [shared_genes[i] for i in keep_in_shared]
                centroids_use = centroids_shared[:, keep_in_shared]
            else:
                use_genes = list(shared_genes); centroids_use = centroids_shared
        n_genes_use = len(use_genes)

        # flat transcript arrays for use_genes → GPU
        all_x, all_y, all_z, all_g = [], [], [], []
        for gidx_use, g in enumerate(use_genes):
            locs = np.asarray(grouped_map[g], dtype=np.float64)
            if locs.shape[0] == 0:
                continue
            all_x.append(locs[:, 0]); all_y.append(locs[:, 1])
            all_z.append(locs[:, 2] if locs.shape[1] >= 3
                         else np.zeros(locs.shape[0], dtype=np.float64))
            all_g.append(np.full(locs.shape[0], gidx_use, dtype=np.int64))
        if all_x:
            loc_x_np = np.concatenate(all_x); loc_y_np = np.concatenate(all_y)
            loc_z_np = np.concatenate(all_z); gi_np = np.concatenate(all_g)
        else:
            loc_x_np = loc_y_np = loc_z_np = np.zeros(0, np.float64)
            gi_np = np.zeros(0, np.int64)
        cx, n_chunk = self.calc_chunk(vx, vy, vz, n_genes_use, mode='map_celltypes_cpu')
        if cx is None or cx <= 0:
            raise RuntimeError('Cannot determine chunk size.')
        return dict(
            vx=vx, vy=vy, vz=vz, dim=(2 if vz == 1 else 3),
            sd_um=float(sampling_distance), bandwidth=float(bandwidth),
            prune_coefficient=float(prune_coefficient),
            norm_threshold=norm_threshold,
            size_after_normalization=float(size_after_normalization),
            n_genes_use=int(n_genes_use), genes=genes, use_genes=use_genes,
            centroids_use=centroids_use, cx=int(cx), n_chunk=int(n_chunk),
            loc_x_t=torch.as_tensor(loc_x_np, dtype=torch.float64, device=gpu_device),
            loc_y_t=torch.as_tensor(loc_y_np, dtype=torch.float64, device=gpu_device),
            loc_z_t=torch.as_tensor(loc_z_np, dtype=torch.float64, device=gpu_device),
            gi_t=torch.as_tensor(gi_np, dtype=torch.int64, device=gpu_device),
            gpu_device=gpu_device,
        )

    def _kde_field_iter_gpu(self, st, *, chunk_start, chunk_end, log_transform,
                            want_full_field, query_subchunk_size, gpu_target_bytes):
        """PRODUCER. Yield, per output x-slab, a dict with the vf_norm plane, the
        KEPT (vf_norm>threshold) voxels' local positions + normalized·log1p gene
        vectors (GPU tensor), and the full normalized field (only if
        ``want_full_field``). KDE math is verbatim from the legacy gpu_core loop."""
        import torch
        from tqdm.auto import tqdm
        from ._mapping_gpu import _kde_setup_celllist, _kde_run_subchunk
        vx, vy, vz = st['vx'], st['vy'], st['vz']
        n_genes_use = st['n_genes_use']; sd_um = st['sd_um']
        bandwidth = st['bandwidth']; prune_coefficient = st['prune_coefficient']
        dim = st['dim']; norm_threshold = st['norm_threshold']
        san = st['size_after_normalization']; gpu_device = st['gpu_device']
        loc_x_t, loc_y_t, loc_z_t, gi_t = st['loc_x_t'], st['loc_y_t'], st['loc_z_t'], st['gi_t']
        cx = st['cx']; n_chunk = st['n_chunk']
        if query_subchunk_size is None:
            query_subchunk_size = max(1024, int(gpu_target_bytes // (n_genes_use * 4)))
        cs = 0 if chunk_start is None else chunk_start
        ce = n_chunk if chunk_end is None else chunk_end
        for chunk_idx in range(cs, ce):
            x0 = chunk_idx * cx; x1 = min(vx, x0 + cx); local_cx = x1 - x0
            qx_grid = (np.arange(x0, x1) * sd_um).astype(np.float64)
            qy_grid = (np.arange(vy) * sd_um).astype(np.float64)
            qz_grid = (np.arange(vz) * sd_um).astype(np.float64)
            Qx, Qy, Qz = np.meshgrid(qx_grid, qy_grid, qz_grid, indexing='ij')
            qx_flat = Qx.ravel(); qy_flat = Qy.ravel(); qz_flat = Qz.ravel()
            N_total = int(qx_flat.size)

            vf_norm_chunk = np.zeros((local_cx, vy, vz), dtype=np.float32)
            full_chunk = (np.zeros((n_genes_use, local_cx, vy, vz), dtype=np.float32)
                          if want_full_field else None)
            keep_x, keep_y, keep_z, keep_vecs = [], [], [], []

            cqxmin = float(qx_grid.min()) if N_total > 0 else 0.0
            cqxmax = float(qx_grid.max()) if N_total > 0 else 0.0
            cqymin = float(qy_grid.min()) if N_total > 0 else 0.0
            cqymax = float(qy_grid.max()) if N_total > 0 else 0.0
            cqzmin = float(qz_grid.min()) if N_total > 0 else 0.0
            cqzmax = float(qz_grid.max()) if N_total > 0 else 0.0
            _Rw_pad = float(bandwidth * prune_coefficient + sd_um)
            mask_chunk_t = (loc_x_t >= cqxmin - _Rw_pad) & (loc_x_t <= cqxmax + _Rw_pad)
            state = _kde_setup_celllist(
                loc_x_t[mask_chunk_t], loc_y_t[mask_chunk_t], loc_z_t[mask_chunk_t],
                gi_t[mask_chunk_t], cqxmin, cqymin, cqzmin, cqxmax, cqymax, cqzmax,
                sd_um, bandwidth, prune_coefficient, dim)
            del mask_chunk_t

            n_sub = (N_total + query_subchunk_size - 1) // query_subchunk_size
            sub_iter = range(n_sub)
            if self.verbose:
                sub_iter = tqdm(sub_iter, desc=f'KDE chunk {chunk_idx + 1}/{n_chunk}',
                                leave=True, dynamic_ncols=True)
            for s in sub_iter:
                s0 = s * query_subchunk_size; s1 = min(N_total, s0 + query_subchunk_size)
                kde_t = _kde_run_subchunk(state, qx_flat[s0:s1], qy_flat[s0:s1],
                                          qz_flat[s0:s1], n_genes_use, gpu_device)
                vf_norm_t = kde_t.sum(dim=1)
                sub_idx = np.arange(s0, s1)
                local_x = sub_idx // (vy * vz); rem = sub_idx % (vy * vz)
                local_y = rem // vz; local_z = rem % vz
                vf_norm_chunk[local_x, local_y, local_z] = (
                    vf_norm_t.detach().cpu().numpy().astype(np.float32, copy=False))
                if want_full_field:
                    vf_full_t = (kde_t / torch.clamp(vf_norm_t, min=1e-12).unsqueeze(1)
                                 * float(san))
                    if log_transform:
                        vf_full_t = torch.log1p(vf_full_t)
                    full_chunk[:, local_x, local_y, local_z] = (
                        vf_full_t.T.detach().cpu().numpy().astype(np.float32, copy=False))
                    del vf_full_t
                mask_t = vf_norm_t > norm_threshold
                if int(mask_t.sum().item()) == 0:
                    del kde_t, vf_norm_t, mask_t
                    continue
                data_kept_t = kde_t[mask_t]
                vfn_kept_t = vf_norm_t[mask_t]
                data_kept_t = data_kept_t / vfn_kept_t.unsqueeze(1) * float(san)
                if log_transform:
                    data_kept_t = torch.log1p(data_kept_t)
                mask_np = mask_t.detach().cpu().numpy()
                keep_x.append(local_x[mask_np]); keep_y.append(local_y[mask_np])
                keep_z.append(local_z[mask_np]); keep_vecs.append(data_kept_t)
                del kde_t, vf_norm_t, mask_t, vfn_kept_t
            kx = np.concatenate(keep_x) if keep_x else np.zeros(0, np.int64)
            ky = np.concatenate(keep_y) if keep_y else np.zeros(0, np.int64)
            kz = np.concatenate(keep_z) if keep_z else np.zeros(0, np.int64)
            vecs = torch.cat(keep_vecs, dim=0) if keep_vecs else None
            yield dict(x0=x0, x1=x1, vf_norm=vf_norm_chunk, full=full_chunk,
                       kx=kx, ky=ky, kz=kz, vecs=vecs)
            del Qx, Qy, Qz, qx_flat, qy_flat, qz_flat, state
            gc.collect(); torch.cuda.empty_cache()

    def _transcript_gpu(self, sdata, *, do_map, save_vf, centroids=None,
                        element_name=None, exclude_gene_indices=None,
                        bandwidth=2.5, sampling_distance=1.0, prune_coefficient=4.3,
                        size_after_normalization=10, log_transform=True, overwrite=True,
                        gpu_device='cuda', query_subchunk_size=None,
                        gpu_target_bytes=int(1e9), chunk_start=None, chunk_end=None,
                        cx=None, create_arrays=True, attach=True):
        """ORCHESTRATOR. Drive the KDE producer into the mapping consumer and/or a
        tensorstore field writer in ONE pass. Covers all transcript combos:
          map_celltypes save_vf=False → do_map=True,  save_vf=False (stream+map)
          map_celltypes save_vf=True  → do_map=True,  save_vf=True  (map+store, 1 pass)
          compute_vf(imaging)         → do_map=False, save_vf=True  (store only)
        """
        import torch
        from ._mapping_gpu import _pearson_max_gpu
        from ._count import _tensorstore_array
        if not torch.cuda.is_available():
            raise RuntimeError('transcript map/compute_vf GPU path requires CUDA')
        st = self._prepare_transcript_field(
            sdata, centroids=centroids, element_name=element_name,
            exclude_gene_indices=exclude_gene_indices, bandwidth=bandwidth,
            sampling_distance=sampling_distance, prune_coefficient=prune_coefficient,
            size_after_normalization=size_after_normalization, gpu_device=gpu_device)
        vx, vy, vz = st['vx'], st['vy'], st['vz']; n_genes_use = st['n_genes_use']
        zg = self.dataset.zarr_group

        def _safe_remove(name):
            if name in zg:
                del zg[name]

        if cx is not None:
            st['cx'] = int(cx)
            st['n_chunk'] = (vx + int(cx) - 1) // int(cx)

        # ---- array setup ----
        if create_arrays:
            if overwrite:
                for nm in ('vf_norm', 'celltype_maps', 'max_correlations',
                           'vf_normalized', 'vf_normalized_genes'):
                    _safe_remove(nm)
            _safe_remove('genes'); zg['genes'] = np.asarray(st['genes'])
            self.dataset._zarr_create_array(zg, 'vf_norm', (vx, vy, vz), 'f4')
            if do_map:
                self.dataset._zarr_create_array(zg, 'celltype_maps', (vx, vy, vz), 'i4')
                self.dataset._zarr_create_array(zg, 'max_correlations', (vx, vy, vz), 'f4')
        # vf_normalized writer (tensorstore, zarr fallback)
        vf_ts = None; vf_zarr = None
        if save_vf and create_arrays:
            _safe_remove('vf_normalized'); _safe_remove('vf_normalized_genes')
            zg['vf_normalized_genes'] = np.asarray(st['use_genes'])
            store_root = getattr(zg.store, 'root', None)
            group_dir = (os.path.join(str(store_root), zg.path) if zg.path
                         else str(store_root)) if store_root is not None else None
            shape = (n_genes_use, vx, vy, vz); chunks = (n_genes_use, 1, vy, vz)
            vf_ts = _tensorstore_array(group_dir, 'vf_normalized', shape, chunks)
            if vf_ts is None:
                vf_zarr = zg.create_array('vf_normalized', shape=shape, chunks=chunks,
                                          dtype='f4', fill_value=0.0)
        elif save_vf:                                  # distributed: write pre-created array
            vf_zarr = zg['vf_normalized']

        centroids_t = (torch.as_tensor(st['centroids_use'], dtype=torch.float32,
                                       device=gpu_device) if do_map else None)
        inflight = []
        it = self._kde_field_iter_gpu(
            st, chunk_start=chunk_start, chunk_end=chunk_end, log_transform=log_transform,
            want_full_field=save_vf, query_subchunk_size=query_subchunk_size,
            gpu_target_bytes=gpu_target_bytes)
        for fc in it:
            x0, x1 = fc['x0'], fc['x1']
            zg['vf_norm'][x0:x1, :, :] = fc['vf_norm']
            if save_vf and fc['full'] is not None:
                if vf_ts is not None:
                    inflight.append(vf_ts[:, x0:x1, :, :].write(fc['full']))
                    while len(inflight) > 4:
                        inflight.pop(0).result()
                else:
                    vf_zarr[:, x0:x1, :, :] = fc['full']
            if do_map:
                local_cx = x1 - x0
                ctmap = np.full((local_cx, vy, vz), -1, dtype=np.int32)
                corr = np.full((local_cx, vy, vz), -1.0, dtype=np.float32)
                if fc['vecs'] is not None and fc['vecs'].shape[0] > 0:
                    best_idx, best_corr = _pearson_max_gpu(centroids_t, fc['vecs'])
                    ctmap[fc['kx'], fc['ky'], fc['kz']] = best_idx.cpu().numpy()
                    corr[fc['kx'], fc['ky'], fc['kz']] = best_corr.cpu().numpy()
                zg['celltype_maps'][x0:x1, :, :] = ctmap
                zg['max_correlations'][x0:x1, :, :] = corr
        for f in inflight:
            f.result()
        if attach:
            import dask.array as _da
            self.dataset._vf_norm = _da.from_zarr(zg['vf_norm'])
            if do_map:
                self.dataset.celltype_maps = zg['celltype_maps'][:]
                self.dataset.max_correlations = zg['max_correlations'][:]

    # ------------------------------------------------------------------
    # Distributed map_celltypes_cpu helpers
    # ------------------------------------------------------------------

    def init_map_celltypes(self, sdata, centroids=None, element_name=None,
                               exclude_gene_indices=None,
                               bandwidth=2.5, sampling_distance=1.0, prune_coefficient=4.3,
                               size_after_normalization=10, log_transform=True):
        """
        Initialize zarr arrays for a distributed ``map_celltypes_cpu`` run.

        Must be called **once** on any single machine before launching
        parallel workers.  After this returns, call
        ``map_celltypes_partial`` on each worker with the returned
        chunk info.

        Parameters
        ----------
        sdata : spatialdata.SpatialData
        centroids : ndarray (n_clusters, n_genes), optional
            Falls back to ``self.dataset.centroids``.
        bandwidth, sampling_distance, prune_coefficient,
        size_after_normalization, log_transform
            Same as ``map_celltypes_cpu``; recorded so workers can use
            identical values.

        Returns
        -------
        dict
            ``{'cx': int, 'n_chunk': int, 'vx': int, 'vy': int, 'vz': int,
               'bandwidth': float, 'sampling_distance': float,
               'prune_coefficient': float,
               'size_after_normalization': float, 'log_transform': bool}``
        """
        # ---- load zarr state ----
        self.dataset.cluster_labels          = self.dataset.zarr_group['cluster_labels'][:]
        self.dataset.downsampled_vectors              = self.dataset.zarr_group['downsampled_vectors'][:]
        self.dataset.normalized_vectors      = self.dataset.zarr_group['normalized_vectors'][:]
        self.dataset.filtered_cluster_labels = self.dataset.zarr_group['filtered_cluster_labels'][:]
        self.dataset.centroids               = self.dataset.zarr_group['centroids'][:]
        self.dataset.centroids_stdev         = self.dataset.zarr_group['centroids_stdev'][:]

        # ---- centroids ----
        if centroids is None:
            if getattr(self.dataset, 'centroids', None) is None:
                raise ValueError('centroids not found. Run cluster_vectors() first.')
            centroids = np.asarray(self.dataset.centroids, dtype=np.float32)
        else:
            centroids = np.asarray(centroids, dtype=np.float32)

        # ---- grid ----
        if getattr(self.dataset, 'shape', None) is None or \
                not hasattr(self, 'locations') or self.locations is None:
            self._prepare_grid(sdata, element_name=element_name,
                               bandwidth=bandwidth,
                               sampling_distance=sampling_distance,
                               prune_coefficient=prune_coefficient,
                               size_after_normalization=size_after_normalization)

        vx, vy, vz = (int(self.dataset.shape[0]),
                      int(self.dataset.shape[1]),
                      int(self.dataset.shape[2]))

        self.dataset.bandwidth = bandwidth
        self.dataset.ndim = 2 if vz == 1 else 3

        # ---- gene list ----
        grouped = self.locations.groupby('gene', sort=True)
        grouped_map = {gene: loc for gene, loc in grouped}

        if getattr(self.dataset, 'genes', None) is None:
            genes = sorted(grouped_map.keys())
            self.dataset.genes = genes
        else:
            genes = list(self.dataset.genes)

        if centroids.shape[1] != len(genes):
            raise ValueError(
                f'centroids gene dim ({centroids.shape[1]}) != gene count ({len(genes)})'
            )

        shared_gene_indices = [i for i, g in enumerate(genes) if g in grouped_map]
        shared_genes = [genes[i] for i in shared_gene_indices]
        if len(shared_genes) == 0:
            raise ValueError('No common genes between input data and centroids.')

        # ---- zarr init ----
        def _safe_remove(name):
            if name in self.dataset.zarr_group:
                del self.dataset.zarr_group[name]

        for name in ['vf_norm', 'celltype_maps', 'max_correlations']:
            _safe_remove(name)

        import numcodecs as _numcodecs
        _safe_remove('genes')
        self.dataset.zarr_group['genes'] = np.asarray(genes)

        self.dataset._zarr_create_array(self.dataset.zarr_group, 'vf_norm', (vx, vy, vz), 'f4')
        self.dataset._zarr_create_array(self.dataset.zarr_group, 'celltype_maps', (vx, vy, vz), 'i4')
        self.dataset._zarr_create_array(self.dataset.zarr_group, 'max_correlations', (vx, vy, vz), 'f4')

        # ---- chunk info ----
        cx, n_chunk = self.calc_chunk(vx, vy, vz, len(shared_genes), mode='map_celltypes_cpu')
        if cx is None or cx <= 0:
            raise RuntimeError('Cannot determine chunk size.')

        if self.verbose:
            print(f'init_map_celltypes: voxel grid ({vx}, {vy}, {vz}), '
                  f'cx={cx}, n_chunk={n_chunk}, shared_genes={len(shared_genes)}')

        return {
            'cx': cx,
            'n_chunk': n_chunk,
            'vx': vx, 'vy': vy, 'vz': vz,
            'bandwidth': bandwidth,
            'sampling_distance': sampling_distance,
            'prune_coefficient': prune_coefficient,
            'size_after_normalization': size_after_normalization,
            'log_transform': log_transform,
        }

    def map_celltypes_partial(self, sdata, chunk_start, chunk_end, cx,
                                  centroids=None, element_name=None,
                                  exclude_gene_indices=None,
                                  bandwidth=2.5, sampling_distance=1.0, prune_coefficient=4.3,
                                  size_after_normalization=10, log_transform=True,
                                  gpu_device=None, force_cpu=False,
                                  query_subchunk_size=None, gpu_target_bytes=int(1e9)):
        """
        Process a contiguous range of X-axis chunks for a distributed run.

        Call ``init_map_celltypes`` once first to create zarr arrays and
        obtain the shared ``cx`` / ``n_chunk`` values.  Then call this
        function on each worker with non-overlapping ``[chunk_start,
        chunk_end)`` ranges.

        Each worker runs on **GPU** when ``gpu_device`` is given and a CUDA
        device is available (and ``force_cpu`` is False); otherwise it
        falls back to the CPU implementation. GPU and CPU paths share the
        same I/O contract — both write the zarr slices ``[x0:x1, :, :]``
        for the chunks they own — so workers may even mix backends.

        Parameters
        ----------
        sdata : spatialdata.SpatialData
            Used to populate ``self.locations`` via ``_prepare_grid``.
        chunk_start : int
            First chunk index to process (inclusive).
        chunk_end : int
            Last chunk index to process (exclusive).
        cx : int
            Chunk width in voxels along the X axis.  **Must be the same
            value across all workers.**  Obtain from
            ``init_map_celltypes`` return value.
        gpu_device : str, optional
            CUDA device for this worker (e.g. ``'cuda:0'``). If None, the
            worker uses the CPU path.
        force_cpu : bool, default False
            Force the CPU path even if ``gpu_device`` is given.
        query_subchunk_size, gpu_target_bytes
            Forwarded to the GPU core (ignored on CPU).
        bandwidth, sampling_distance, prune_coefficient,
        size_after_normalization, log_transform, centroids,
        element_name, exclude_gene_indices
            Same as ``map_celltypes_cpu``.  Values must match those used
            during ``init_map_celltypes``.

        Returns
        -------
        None
            Results are written to the zarr store slices
            ``[x0:x1, :, :]`` for each processed chunk.
        """
        # ---- GPU dispatch: per-worker chunk range on the supplied device ----
        use_gpu = False
        if gpu_device is not None and not force_cpu:
            try:
                from ._kde_gpu import gpu_available
                use_gpu = gpu_available(gpu_device)
            except Exception:
                use_gpu = False
        if use_gpu:
            self._m(f'map_celltypes_partial: GPU path ({gpu_device}) '
                    f'chunks {chunk_start}..{chunk_end - 1}')
            return self._transcript_gpu(
                sdata, do_map=True, save_vf=False,
                chunk_start=chunk_start, chunk_end=chunk_end, cx=cx,
                create_arrays=False, attach=False,
                centroids=centroids, element_name=element_name,
                exclude_gene_indices=exclude_gene_indices, bandwidth=bandwidth,
                sampling_distance=sampling_distance, prune_coefficient=prune_coefficient,
                size_after_normalization=size_after_normalization,
                log_transform=log_transform, overwrite=False,
                gpu_device=gpu_device, query_subchunk_size=query_subchunk_size,
                gpu_target_bytes=gpu_target_bytes,
            )

        from tqdm.auto import tqdm

        # ---- load zarr state ----
        self.dataset.cluster_labels          = self.dataset.zarr_group['cluster_labels'][:]
        self.dataset.downsampled_vectors              = self.dataset.zarr_group['downsampled_vectors'][:]
        self.dataset.normalized_vectors      = self.dataset.zarr_group['normalized_vectors'][:]
        self.dataset.filtered_cluster_labels = self.dataset.zarr_group['filtered_cluster_labels'][:]
        self.dataset.centroids               = self.dataset.zarr_group['centroids'][:]
        self.dataset.centroids_stdev         = self.dataset.zarr_group['centroids_stdev'][:]

        # ---- centroids ----
        if centroids is None:
            if getattr(self.dataset, 'centroids', None) is None:
                raise ValueError('centroids not found. Run cluster_vectors() first.')
            centroids = np.asarray(self.dataset.centroids, dtype=np.float32)
        else:
            centroids = np.asarray(centroids, dtype=np.float32)

        # ---- grid / locations ----
        if getattr(self.dataset, 'shape', None) is None or \
                not hasattr(self, 'locations') or self.locations is None:
            self._prepare_grid(sdata, element_name=element_name,
                               bandwidth=bandwidth,
                               sampling_distance=sampling_distance,
                               prune_coefficient=prune_coefficient,
                               size_after_normalization=size_after_normalization)

        vx, vy, vz = (int(self.dataset.shape[0]),
                      int(self.dataset.shape[1]),
                      int(self.dataset.shape[2]))

        self.dataset.bandwidth = bandwidth
        self.dataset.ndim = 2 if vz == 1 else 3
        self.dataset.expression_threshold = (
            1 / (np.sqrt(2 * np.pi) * bandwidth) ** self.dataset.ndim
        )
        self.dataset.norm_threshold = self.dataset.expression_threshold * 2

        # ---- gene list ----
        grouped = self.locations.groupby('gene', sort=True)
        grouped_map = {gene: loc for gene, loc in grouped}

        if getattr(self.dataset, 'genes', None) is None:
            genes = sorted(grouped_map.keys())
            self.dataset.genes = genes
        else:
            genes = list(self.dataset.genes)

        if centroids.shape[1] != len(genes):
            raise ValueError(
                f'centroids gene dim ({centroids.shape[1]}) != gene count ({len(genes)})'
            )

        shared_gene_indices = [i for i, g in enumerate(genes) if g in grouped_map]
        shared_genes = [genes[i] for i in shared_gene_indices]
        missing_genes = [g for g in genes if g not in grouped_map]
        if len(shared_genes) == 0:
            raise ValueError('No common genes between input data and centroids.')
        if missing_genes and self.verbose:
            print(f'map_celltypes_partial: dropping {len(missing_genes)} missing genes '
                  f'(e.g. {missing_genes[:5]})')

        grouped_list = [(g, grouped_map[g]) for g in shared_genes]
        centroids_shared = centroids[:, shared_gene_indices]

        if exclude_gene_indices is not None:
            idx_to_local = {gi: li for li, gi in enumerate(shared_gene_indices)}
            exclude_shared = sorted(
                idx_to_local[i] for i in exclude_gene_indices if i in idx_to_local
            )
            centroids_use = np.delete(centroids_shared, exclude_shared, axis=1)
        else:
            exclude_shared = None
            centroids_use = centroids_shared

        import numcodecs as _nc2
        _nc2.blosc.set_nthreads(self.ncores)

        n_chunks_total = chunk_end - chunk_start
        if self.verbose:
            print(f'map_celltypes_partial: processing chunks '
                  f'{chunk_start}..{chunk_end - 1} (cx={cx}, vx={vx})')

        # ---- chunk loop ----
        for local_idx, chunk_idx in enumerate(range(chunk_start, chunk_end)):
            x0 = chunk_idx * cx
            x1 = min(vx, x0 + cx)
            local_cx = x1 - x0

            vf_chunk = np.zeros((local_cx, vy, vz, len(shared_genes)), dtype='f4')

            if self.verbose:
                gene_bar = tqdm(
                    total=len(grouped_list),
                    desc=f'KDE genes chunk {chunk_idx} ({local_idx + 1}/{n_chunks_total})',
                    leave=True, dynamic_ncols=True,
                )
            else:
                gene_bar = None

            for gidx, (_, loc) in enumerate(grouped_list):
                locs = np.asarray(loc)
                if locs.shape[1] == 2:
                    loc_x, loc_y = locs[:, 0].astype(np.float64), locs[:, 1].astype(np.float64)
                    loc_z = np.zeros(locs.shape[0], dtype=np.float64)
                    dim = 2
                else:
                    loc_x, loc_y, loc_z = (locs[:, 0].astype(np.float64),
                                           locs[:, 1].astype(np.float64),
                                           locs[:, 2].astype(np.float64))
                    dim = 3

                accumulate_gene_vector(
                    vf_chunk,
                    loc_x=loc_x, loc_y=loc_y, loc_z=loc_z,
                    sampling_distance=float(sampling_distance),
                    bandwidth=float(bandwidth),
                    prune_coefficient=float(prune_coefficient),
                    gidx=gidx, dim=int(dim), ncores=int(self.ncores),
                    x0=int(x0), y0=0, z0=0,
                )
                if gene_bar is not None:
                    gene_bar.update(1)

            if gene_bar is not None:
                gene_bar.close()

            vf_norm_chunk = vf_chunk.sum(axis=3)
            self.dataset.zarr_group['vf_norm'][x0:x1, :, :] = vf_norm_chunk

            local_ctmap = np.full((local_cx, vy, vz), -1, dtype=np.int32)
            local_corr = np.full((local_cx, vy, vz), -1, dtype=np.float32)

            x_idx, y_idx, z_idx, data_chunk = normalize_vf(
                vf_chunk, vf_norm_chunk, size_after_normalization, x0, self.ncores
            )
            if log_transform:
                np.log1p(data_chunk, out=data_chunk)

            if data_chunk.shape[0] > 0:
                data_use = (np.delete(data_chunk, exclude_shared, axis=1)
                            if exclude_shared is not None else data_chunk)
                best_corr = np.full(data_use.shape[0], -1, dtype=np.float32)
                best_idx = np.full(data_use.shape[0], -1, dtype=np.int32)

                if self.verbose:
                    corr_bar = tqdm(
                        total=len(centroids_use),
                        desc=f'Correlations chunk {chunk_idx} ({local_idx + 1}/{n_chunks_total})',
                        leave=True, dynamic_ncols=True,
                    )
                else:
                    corr_bar = None

                for cidx, centroid in enumerate(centroids_use):
                    corr_chunk = calc_ctmap(centroid, data_use, self.ncores)
                    corr_chunk = np.nan_to_num(corr_chunk).astype(np.float32, copy=False)
                    improved = corr_chunk > best_corr
                    best_corr[improved] = corr_chunk[improved]
                    best_idx[improved] = cidx
                    if corr_bar is not None:
                        corr_bar.update(1)

                if corr_bar is not None:
                    corr_bar.close()

                local_x = x_idx.astype(np.int64) - x0
                local_y = y_idx.astype(np.int64)
                local_z = z_idx.astype(np.int64)
                local_ctmap[local_x, local_y, local_z] = best_idx
                local_corr[local_x, local_y, local_z] = best_corr

            self.dataset.zarr_group['celltype_maps'][x0:x1, :, :] = local_ctmap
            self.dataset.zarr_group['max_correlations'][x0:x1, :, :] = local_corr

            del vf_chunk, vf_norm_chunk, x_idx, y_idx, z_idx, data_chunk
            del local_ctmap, local_corr
            gc.collect()

        if self.verbose:
            print(f'map_celltypes_partial: done chunks {chunk_start}..{chunk_end - 1}')
