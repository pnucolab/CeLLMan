import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import seaborn as sns
import multiprocessing
import sys, os
sns.set()
sns.set_style("whitegrid", {'axes.grid' : False})
from sklearn.decomposition import PCA
from tempfile import mkdtemp
from matplotlib.colors import ListedColormap 
from sklearn import preprocessing
import numbers
import zarr
import dask.array as da 

# Try to import xarray and SpatialData for image saving
try:
    import xarray as xr
    from spatialdata.models import Image2DModel, Labels2DModel, TableModel
    from spatialdata.transformations import Affine
    from spatialdata.transformations.operations import get_transformation, set_transformation
    _SPATIALDATA_AVAILABLE = True
except ImportError:
    _SPATIALDATA_AVAILABLE = False

# Try to import utils module (C++ extension)
try:
    from .utils import corr
except ImportError:
    # Fallback to pure Python implementation
    import numpy as np
    corr = lambda a, b: np.corrcoef(a, b)[0, 1]

class SSAMDataset(object):
    """
    A class to store intial values and results of SSAM analysis.

    :param genes: The genes that will be used for the analysis.
    :type genes: list(str)    
    :param locations: Location of the mRNAs in um, given as a list of
        N x D ndarrays (N is number of mRNAs, D is number of dimensions).
    :type locations: list(numpy.ndarray)
    :param width: Width of the image in um.
    :type width: float
    :param height: Height of the image in um.
    :type height: float
    :param depth: Depth of the image in um. Depth == 1 means 2D image.
    :type depth: float
    :param save_dir: Directory to store intermediate data as zarr groups (e.g. density / vector field).
        Any data which already exists will be loaded and reused.
    :type save_dir: str
    """
        
    def __init__(self, store=None, in_memory=False):
        self._vf = None
        self._vf_norm = None
        self._vf_normalized = None
        self.bandwidth = None
        self._downsampled_vectors = None
        self._downsampled_vf_norm = None
        self._selected_vectors = None
        self.normalized_vectors = None
        self.expanded_vectors = None
        self.cluster_labels = None
        #self.corr_map = None
        self.tsne = None
        self.umap = None
        self.excluded_clusters = None
        self.celltype_binned_counts = None
        self.max_probabilities = None
        self.zarr_store, self.zarr_group = self._get_zarr_group(store)
        self.in_memory = in_memory
        
        # Initialize scale properties for SpatialData coordinate transformations
        self.x_scale = None
        self.y_scale = None
    
    @staticmethod
    def _get_zarr_group(store):
        if store is None:
            # memory store
            return None, zarr.group()

        zarr_format = None
        if isinstance(store, str):
            # Detect existing zarr format from on-disk markers so we
            # don't accidentally open a v2-written store in v3 mode
            # (which would hide all the v2 arrays already in it). zarr 3
            # warns about this case but still picks v3 by default.
            import os as _os
            store_path = store
            has_v2 = _os.path.exists(_os.path.join(store_path, '.zgroup'))
            has_v3 = _os.path.exists(_os.path.join(store_path, 'zarr.json'))
            if has_v2 and not has_v3:
                zarr_format = 2
            elif has_v3 and not has_v2:
                zarr_format = 3
            elif has_v2 and has_v3:
                # Mixed store: prefer v2 since most legacy ssam state
                # was written before the zarr 3 migration. Recently
                # written v3 arrays (e.g., vf_norm, max_correlations)
                # are not visible in this mode and need re-writing.
                zarr_format = 2

            # zarr 3+: zarr.storage.LocalStore; zarr 2: zarr.DirectoryStore
            if hasattr(zarr, 'storage') and hasattr(zarr.storage, 'LocalStore'):
                store = zarr.storage.LocalStore(store_path)
            else:
                store = zarr.DirectoryStore(store_path)

        open_kwargs = dict(store=store, mode="a")
        if zarr_format is not None:
            open_kwargs['zarr_format'] = zarr_format
        return store, zarr.open_group(**open_kwargs)

    @staticmethod
    def _zarr_create_array(group, name, shape, dtype):
        """Create a zero-filled array inside ``group``, respecting the
        group's zarr_format.

        zarr 3's ``Group.zeros(...)`` always writes v3 metadata
        (``zarr.json``) regardless of the parent group's format, which
        breaks lookups when the group is opened in v2 mode. This helper
        prefers ``create_array`` (which respects ``group.metadata``) and
        falls back to ``create_dataset`` for legacy zarr 2.
        """
        shape = tuple(shape) if hasattr(shape, '__iter__') else (int(shape),)
        if hasattr(group, 'create_array'):
            return group.create_array(name=name, shape=shape, dtype=dtype)
        return group.create_dataset(name=name, shape=shape, dtype=dtype)

    @staticmethod
    def _zarr_store_path(store):
        """Return the on-disk path of a zarr store, compatible with zarr 2/3."""
        if store is None:
            return None
        # zarr 3 LocalStore exposes .root; zarr 2 DirectoryStore exposes .path
        for attr in ('root', 'path'):
            p = getattr(store, attr, None)
            if p is not None:
                return str(p)
        return None
    
    def _try_flush(self):
        if self.zarr_store is not None:
            try:
                self.zarr_store.flush()
            except:
                pass
    
    def extract_spatialdata_scale(self, sdata, element_name='transcripts'):
        """
        Extract coordinate transformation scale from SpatialData for proper coordinate mapping.
        
        :param sdata: SpatialData object containing points data
        :type sdata: spatialdata.SpatialData
        :param element_name: Name of the points element to extract scale from
        :type element_name: str
        :returns: Tuple of (x_scale, y_scale) or None if extraction fails
        :rtype: tuple or None
        """
        if not _SPATIALDATA_AVAILABLE:
            print("Warning: SpatialData not available. Cannot extract scale information.")
            return None
            
        try:
            # Find the specified points element
            for element_type, elem_name, element in list(sdata.gen_elements()):
                if element_type == 'points' and elem_name == element_name:
                    transforms = get_transformation(element, get_all=True)
                    t = transforms['global']
                    matrix = t.to_affine_matrix(input_axes=('x', 'y'), output_axes=('x', 'y'))
                    self.x_scale = matrix[0, 0]
                    self.y_scale = matrix[1, 1]
                    # Keep the full element→global affine (2x3) so the overlay can
                    # apply translation/rotation, not just the diagonal scale.
                    # Some platforms (e.g. MERSCOPE) carry a non-zero translation
                    # (y-shift) on the transcripts transform; dropping it offsets
                    # the downsampled-vector scatter from the morphology image.
                    self._global_affine = np.asarray(matrix[:2, :], dtype=np.float64)
                    return (self.x_scale, self.y_scale)
            
            print(f"Warning: Points element '{element_name}' not found in SpatialData.")
            return None
            
        except Exception as e:
            print(f"Warning: Failed to extract scale from SpatialData: {e}")
            return None
    
    @property
    def downsampled_vectors(self):
        return self._downsampled_vectors
    
    @downsampled_vectors.setter
    def downsampled_vectors(self, v):
        self._downsampled_vectors = v
        self._selected_vectors = None

    @property
    def downsampled_vf_norm(self):
        """Per-downsampled-vector local KDE density (vf_norm at each sampled
        position), shape (N,), aligned row-for-row with ``downsampled_vectors``
        and ``normalized_vectors``. Computed in ``_select_vectors`` and persisted
        by ``downsample_vector``; used downstream (e.g. ``cluster_vectors`` writes
        it into ``ssam_clusters.h5ad`` obs) as a density signal for background /
        low-density-voxel detection."""
        if self._downsampled_vf_norm is not None:
            return self._downsampled_vf_norm
        if 'downsampled_vf_norm' in self.zarr_group:
            self._downsampled_vf_norm = np.asarray(self.zarr_group['downsampled_vf_norm'][:])
            return self._downsampled_vf_norm
        return None

    @downsampled_vf_norm.setter
    def downsampled_vf_norm(self, v):
        self._downsampled_vf_norm = None if v is None else np.asarray(v)

    @property
    def selected_vectors(self):
        if 'selected_vectors' in self.zarr_group:
            self._selected_vectors = da.from_zarr(self.zarr_group['selected_vectors'])
            return self._selected_vectors
        
        if self._selected_vectors is None:
            assert self._downsampled_vectors is not None
            mask = np.zeros(self.vf_norm.shape, dtype=bool)
            mask[self.downsampled_vectors] = True
            self._selected_vectors = self.vf.reshape([-1, len(self.genes)])[np.ravel(mask)].compute()
        return self._selected_vectors
    
    @selected_vectors.setter
    def selected_vectors(self, v):
        self._selected_vectors = v

    @property
    def vf(self):
        """
        Vector field as a numpy.ndarray.
        """
        return self._vf
    
    @vf.setter
    def vf(self, vf):
        if self.in_memory:
            self._vf = da.array(vf.compute())
        else:
            self._vf = vf
        self._vf_norm = None
        try:
            del self.zarr_group['vf_norm']
        except:
            pass
        
    @property
    def vf_normalized(self):
        return self._vf_normalized
        
    @vf_normalized.setter
    def vf_normalized(self, vf_normalized):
        if self.in_memory:
            self._vf_normalized = da.array(vf_normalized.compute())
        else:
            self._vf_normalized = vf_normalized
            
    # @property
    # def vf_norm(self):
    #     """
    #     `L1-norm <http://mathworld.wolfram.com/L1-Norm.html>`_ of the vector field as a numpy.ndarray.
    #     """

    #     if self.vf is None:
    #         return None
    #     if self._vf_norm is None:
    #         self.zarr_group.zeros(name='vf_norm', shape=self.vf.shape[:-1])
    #         self.zarr_group['vf_norm'] = self.vf.sum(axis=3).compute()
    #         self._try_flush()
    #         self._vf_norm = da.from_zarr(self.zarr_group['vf_norm'])
    #     return self._vf_norm

    @property
    def vf_norm(self):

        if self._vf_norm is not None:
            return self._vf_norm

        if 'vf_norm' in self.zarr_group:
            self._vf_norm = da.from_zarr(self.zarr_group['vf_norm'])
            return self._vf_norm

        if self.vf is None:
            return None

        self.zarr_group.zeros(name='vf_norm', shape=self.vf.shape[:-1])
        self.zarr_group['vf_norm'] = self.vf.sum(axis=3).compute()
        self._try_flush()
        self._vf_norm = da.from_zarr(self.zarr_group['vf_norm'])
        return self._vf_norm

    def get_celltype_correlation(self, idx):
        """
        Get maximum correlation values of a cell type map between the given cluster's centroid to the vector field.

        :param idx: Index of a cluster
        :type idx: int
        :return: Correlation values of a cell type map of the specified cluster's centroid
        :rtype: numpy.ndarray
        """
        rtn = np.zeros_like(self.max_correlations) - 1
        rtn[self.celltype_maps == idx] = self.max_correlations[self.celltype_maps == idx]
        return rtn

    def get_celltype_probability(self, idx):
        """
        Get maximum probability map of a cell type.

        :param idx: Index of a cluster
        :type idx: int
        :return: Maximum probability map
        :rtype: numpy.ndarray
        """
        rtn = np.zeros_like(self.max_probabilities) - 1
        rtn[self.celltype_maps == idx] = self.max_probabilities[self.celltype_maps == idx]
        return rtn


    def _grid_to_original_coords(self, grid_x, grid_y, grid_z=None):
        """KDE 그리드 좌표를 원본 SpatialData 좌표로 변환"""
        if not hasattr(self, 'coordinate_offset') or not hasattr(self, 'sampling_distance'):
            return grid_x, grid_y, grid_z
            
        # 그리드 인덱스 -> 실제 거리(μm) -> 원본 좌표
        real_x = grid_x * self.sampling_distance + self.coordinate_offset[0]
        real_y = grid_y * self.sampling_distance + self.coordinate_offset[1]
        real_z = None
        if grid_z is not None and len(self.coordinate_offset) > 2:
            real_z = grid_z * self.sampling_distance + self.coordinate_offset[2]
            
        return real_x, real_y, real_z
    
    @property
    def downsampled_vectors_original_coords(self):
        """downsampled_vectors를 원본 좌표계로 변환하여 반환"""
        if not hasattr(self, 'downsampled_vectors') or self.downsampled_vectors is None:
            return None
            
        grid_x, grid_y, grid_z = self.downsampled_vectors
        orig_x, orig_y, orig_z = self._grid_to_original_coords(grid_x, grid_y, grid_z)
        
        if orig_z is not None:
            return (orig_x, orig_y, orig_z)
        else:
            return (orig_x, orig_y, grid_z)  # 2D의 경우 z는 그대로

    def _get_plot_extent(self):
        """matplotlib imshow에 사용할 extent 계산"""
        if not hasattr(self, 'original_coordinate_bounds'):
            return None

        x_bounds, y_bounds, z_bounds = self.original_coordinate_bounds
        return [x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]]

    def _plot_celltypes_map(self, data, background="black", centroid_indices=[], colors=None, cmap='jet', rotate=0, min_r=0.6, set_alpha=False, z=None, use_original_coords=True):
        if z is None:
            z = int(self.shape[2] / 2)
        num_ctmaps = np.max(data) + 1

        if len(centroid_indices) == 0:
            centroid_indices = list(range(num_ctmaps))

        if colors is None:
            cmap_internal = plt.get_cmap(cmap)
            colors = cmap_internal([float(i) / (num_ctmaps - 1) for i in range(num_ctmaps)])

        all_colors = [background if not j in centroid_indices else colors[i] for i, j in enumerate(range(num_ctmaps))]
        cmap_internal = ListedColormap(all_colors)

        celltype_maps_internal = np.array(data[..., z].T, copy=True)
        empty_mask = celltype_maps_internal == -1
        celltype_maps_internal[empty_mask] = 0
        sctmap = cmap_internal(celltype_maps_internal)
        sctmap[empty_mask] = (0, 0, 0, 0)

        if set_alpha:
            # per-voxel score = correlation (corr path) or softmax prob (tsvm path)
            score = self.max_correlations if getattr(self, "max_correlations", None) is not None \
                else getattr(self, "max_probabilities", None)
            if score is not None:
                alpha = np.array(score[..., z].T, copy=True)
                alpha[alpha < 0] = 0 # drop negative correlations
                alpha = min_r + alpha / (np.max(alpha) / (1.0 - min_r))
                sctmap[..., 3] = alpha

        if rotate == 1 or rotate == 3:
            sctmap = sctmap.swapaxes(0, 1)


        extent = self._get_plot_extent()
        extent[0] *= self.x_scale
        extent[1] *= self.x_scale
        extent[2], extent[3] = extent[3], extent[2]
        extent[2] *= self.y_scale
        extent[3] *= self.y_scale


        plt.gca().set_facecolor(background)
        plt.imshow(sctmap, interpolation='nearest', extent=extent)

        # plt.gca().invert_yaxis()
        if rotate == 1:
            plt.gca().invert_yaxis()
        elif rotate == 2:
            plt.gca().invert_xaxis()
            plt.gca().invert_yaxis()
        elif rotate == 3:
            plt.gca().invert_xaxis()

    def plot_celltypes_map(self, background="black", centroid_indices=[], colors=None, cmap='jet', rotate=0, min_r=0.6, set_alpha=False, z=None, save_to_spatialdata=True, show_annotation=False, annotation_csv=None, n_celltypes=None):
        """
        Plot the merged cell-type map.

        :param background: Set background color of the cell-type map.
        :type background: str or list(float)
        :param centroid_indices: The centroids which will be in the cell type map. If not given, the cell-type map is drawn with all centroids.
        :type centroid_indices: list(int)
        :param colors: Color of the clusters. Overrides `cmap` parameter.
        :type colors: list(str), list(list(float))
        :param cmap: Colormap for the clusters.
        :type cmap: str or matplotlib.colors.Colormap
        :param rotate: Rotate the plot. Possible values are 0, 1, 2, and 3.
        :type rotate: int
        :param min_r: Minimum correlation threshold for the cell-type map.
            This value is only for the plotting, does not affect to the cell-type maps generated by `filter_celltypemaps`.
        :type min_r: float
        :param set_alpha: Set alpha of each pixel based on the correlation.
            Not properly implemented yet, doesn't work properly with the background other than black.
        :type set_alpha: bool
        :param z: Z index to slice 3D cell-type map.
            If not given, the slice at the middle will be used.
        :type z: int
        :param save_to_spatialdata: Whether to save the celltype map to SpatialData images if input was SpatialData.
        :type save_to_spatialdata: bool
        :param show_annotation: If True and annotation_csv is provided, draw a legend with celltype names.
        :type show_annotation: bool
        :param annotation_csv: Path to annotation CSV (must have 'cluster', 'celltype', optionally 'global_cluster' columns).
        :type annotation_csv: str or None
        :param n_celltypes: Total number of globally unique celltypes. Used with 'global_cluster' column to assign
            consistent colors across tiles. If None, inferred from max(global_cluster) + 1.
        :type n_celltypes: int or None
        """

        # Build per-tile color list for the map.
        # If annotation_csv has global_cluster and no custom colors, use globally consistent colors.
        # This runs regardless of show_annotation so that the map colors are the same whether
        # or not the legend is displayed.
        plot_colors = colors
        ann_df = None
        if annotation_csv is not None and colors is None:
            import pandas as pd
            ann_df = pd.read_csv(annotation_csv)
            if 'global_cluster' in ann_df.columns:
                cmap_internal = plt.get_cmap(cmap)
                # n_global priority: explicit parameter > n_global_celltypes column > max+1 in this CSV.
                # The column-based value is written by cluster-analyst using the full cross-tile count,
                # so it is the same in every tile's CSV and guarantees consistent color mapping.
                if n_celltypes is not None:
                    n_global = n_celltypes
                elif 'n_global_celltypes' in ann_df.columns:
                    n_global = int(ann_df['n_global_celltypes'].iloc[0])
                else:
                    n_global = int(ann_df['global_cluster'].max() + 1)
                num_ctmaps = int(np.max(self.filtered_celltype_maps) + 1)
                # Sort by local 'cluster' to get the same order as filtered_celltype_maps values.
                # exclude_and_merge_clusters preserves order, so rank k → map value k.
                sorted_rows = ann_df.sort_values('cluster')
                plot_colors = [None] * num_ctmaps
                for rank, (_, row) in enumerate(sorted_rows.iterrows()):
                    if rank < num_ctmaps:
                        global_id = int(row['global_cluster'])
                        plot_colors[rank] = cmap_internal(float(global_id) / max(n_global - 1, 1))
                # Fill any unassigned slots with background
                plot_colors = [c if c is not None else background for c in plot_colors]

        self._plot_celltypes_map(self.filtered_celltype_maps, background, centroid_indices, plot_colors, cmap, rotate, min_r, set_alpha, z)

        if show_annotation and annotation_csv is not None:
            import pandas as pd
            from matplotlib.patches import Patch
            if ann_df is None:
                ann_df = pd.read_csv(annotation_csv)
            patches = []
            if colors is None:
                cmap_internal = plt.get_cmap(cmap)
                if 'global_cluster' in ann_df.columns:
                    # Use global_cluster for consistent cross-tile colors (same as map).
                    if n_celltypes is not None:
                        n_global = n_celltypes
                    elif 'n_global_celltypes' in ann_df.columns:
                        n_global = int(ann_df['n_global_celltypes'].iloc[0])
                    else:
                        n_global = int(ann_df['global_cluster'].max() + 1)
                    for _, row in ann_df.sort_values('cluster').iterrows():
                        global_id = int(row['global_cluster'])
                        color = cmap_internal(float(global_id) / max(n_global - 1, 1))
                        patches.append(Patch(facecolor=color, label=row['celltype']))
                else:
                    # Fallback: rank-based coloring (no global_cluster column).
                    # exclude_and_merge_clusters preserves order, so k-th sorted CSV cluster
                    # corresponds to map value k.
                    num_ctmaps = int(np.max(self.filtered_celltype_maps) + 1)
                    for rank, (_, row) in enumerate(ann_df.sort_values('cluster').iterrows()):
                        color = cmap_internal(float(rank) / max(num_ctmaps - 1, 1))
                        patches.append(Patch(facecolor=color, label=row['celltype']))
            else:
                for _, row in ann_df.sort_values('cluster').iterrows():
                    cluster_idx = int(row['cluster'])
                    if cluster_idx < len(colors):
                        patches.append(Patch(facecolor=colors[cluster_idx], label=row['celltype']))
            plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        # Save celltype map to SpatialData if input was SpatialData
        if (save_to_spatialdata and
            _SPATIALDATA_AVAILABLE and
            hasattr(self, '_input_data_type') and
            self._input_data_type == 'spatialdata' and
            hasattr(self, '_original_input_data')):

            self._save_celltype_map_to_spatialdata(
                self.filtered_celltype_maps,
                background,
                centroid_indices,
                colors,
                cmap,
                rotate,
                min_r,
                set_alpha,
                z,
                image_name="celltype_map"
            )

        return

    def write_celltype_map(self, sdata=None, *, colors=None, cmap='jet',
                           background="black", z=None, min_r=0.6, set_alpha=False,
                           annotation_csv=None,
                           labels_key="cellman_celltype_map",
                           image_key="cellman_celltype_map_rgb",
                           table_key="cellman_celltype"):
        """Write the filtered cell-type map into ``sdata`` as three aligned elements.

        * ``labels[labels_key]`` — integer raster; pixel value = ``cluster index +
          1`` (0 = background / Unknown). This is the idiomatic categorical
          (Labels2DModel) representation of the segmentation-free cell-type map.
        * ``images[image_key]`` — an RGB rendering (same colours as
          :meth:`plot_celltypes_map`) for quick visual inspection. (SpatialData
          keys are globally unique across element types, so the RGB image cannot
          share the labels' key; it takes the ``…_rgb`` suffix by default.)
        * ``tables[table_key]`` — annotates the labels: instance id → cell-type
          name (+ colour), built from ``annotation_csv`` when given.

        All three carry the SAME ``Scale(sampling_distance) + Translation(origin)``
        affine into the shared ``'global'`` µm coordinate system, so the map
        overlays the input table / points / staining image exactly.

        ``sdata`` defaults to the SpatialData that was fed to ``compute_vf`` /
        ``downsample_vector`` (remembered as ``_original_input_data``). Returns it.
        """
        if not _SPATIALDATA_AVAILABLE:
            raise ImportError("spatialdata not available — cannot write celltype map")
        sdata = sdata if sdata is not None else getattr(self, "_original_input_data", None)
        if sdata is None:
            raise ValueError("no SpatialData given; pass sdata= or run compute_vf "
                             "with a SpatialData input so it is remembered")

        data = self.filtered_celltype_maps
        if z is None:
            z = int(self.shape[2] / 2)
        lab2d = np.asarray(data[..., z].T)               # (vx=col,vy=row) → (y,x); -1 = bg
        K = int(lab2d.max()) + 1 if (lab2d.size and lab2d.max() >= 0) else 0

        # ---- labels: -1 → 0 background, clusters → 1..K ----
        labels_arr = (lab2d + 1).astype(np.int32)

        # ---- RGB rendering (same colours as plot_celltypes_map) ----
        if colors is None:
            cmap_internal = plt.get_cmap(cmap)
            colors = cmap_internal([float(i) / max(K - 1, 1) for i in range(max(K, 1))])
        lut = ListedColormap([to_rgba(c) for c in colors]) if K else ListedColormap(["black"])
        rgba = lut(np.clip(lab2d, 0, None))
        rgba[lab2d == -1] = (0, 0, 0, 0)
        _score = self.max_correlations if getattr(self, "max_correlations", None) is not None \
            else getattr(self, "max_probabilities", None)
        if set_alpha and _score is not None:
            a = np.asarray(_score[..., z].T, dtype=float); a[a < 0] = 0
            if a.max() > 0:
                rgba[..., 3] = min_r + a / (a.max() / (1.0 - min_r))
        rgb = (rgba[..., :3] * rgba[..., 3:4] * 255).astype(np.uint8)     # premultiply on black
        cyx = np.moveaxis(rgb, -1, 0)                                     # (c, y, x)

        # ---- transform: raster index (x=col, y=row) → world µm ----
        zg = self.zarr_group
        sd_um = float(zg.attrs.get("sampling_distance",
                      (zg["vf_params"][0] if "vf_params" in zg else 1.0)))
        ox, oy = self._celltype_map_origin(sdata)
        # count platforms: x_scale=y_scale=1 → exact Scale(sd)+Translation(origin).
        # transcript platforms fold in the points' world scale (extract_spatialdata_scale).
        xs = float(getattr(self, "x_scale", 1.0) or 1.0)
        ys = float(getattr(self, "y_scale", 1.0) or 1.0)
        mat = np.array([[sd_um * xs, 0.0, ox], [0.0, sd_um * ys, oy], [0.0, 0.0, 1.0]])
        tf = Affine(mat, input_axes=("x", "y"), output_axes=("x", "y"))

        sdata.labels[labels_key] = Labels2DModel.parse(
            labels_arr, dims=("y", "x"), transformations={"global": tf})
        sdata.images[image_key] = Image2DModel.parse(
            cyx, dims=("c", "y", "x"), transformations={"global": tf})
        if annotation_csv is not None:
            self._write_celltype_table(sdata, K, colors, annotation_csv,
                                       labels_key, table_key)
        print(f"[write_celltype_map] labels['{labels_key}'] {labels_arr.shape} "
              f"+ images['{image_key}'] + "
              f"{'tables['+table_key+']' if annotation_csv else 'no table'} "
              f"@ {sd_um}µm/px, origin=({ox:.1f},{oy:.1f})")
        return sdata

    def _celltype_map_origin(self, sdata):
        """World-µm origin ``(x0, y0)`` of the vf grid: count platforms store it in
        the count table's ``uns['cellman']['origin']``; otherwise fall back to the
        dataset ``coordinate_offset``; otherwise ``(0, 0)``."""
        try:
            for t in sdata.tables.values():
                cm = t.uns.get("cellman", {})
                if "origin" in cm:
                    o = cm["origin"]; return float(o[0]), float(o[1])
        except Exception:  # noqa: BLE001
            pass
        off = getattr(self, "coordinate_offset", None)
        if off is not None and len(off) >= 2:
            return float(off[0]), float(off[1])
        return 0.0, 0.0

    def _write_celltype_table(self, sdata, K, colors, annotation_csv,
                              labels_key, table_key):
        """Build the labels-annotating table: instance id (cluster+1) → celltype."""
        import pandas as pd
        from anndata import AnnData
        from matplotlib.colors import to_hex
        names = {}
        try:
            ann = pd.read_csv(annotation_csv)
            if {"cluster", "celltype"}.issubset(ann.columns):
                for _, r in ann.sort_values("cluster").iterrows():
                    names[int(r["cluster"])] = str(r["celltype"])
        except Exception as e:  # noqa: BLE001
            print(f"Warning: could not read annotation_csv ({e}); using cluster ids")
        if K <= 0:
            return
        obs = pd.DataFrame({
            "instance_id": np.arange(1, K + 1, dtype=np.int64),
            "region": pd.Categorical([labels_key] * K),
            "celltype": [names.get(i, f"cluster_{i}") for i in range(K)],
            "color": [to_hex(to_rgba(colors[i])) for i in range(K)],
        })
        tab = AnnData(obs=obs)
        sdata.tables[table_key] = TableModel.parse(
            tab, region=labels_key, region_key="region", instance_key="instance_id")

    def _save_celltype_map_to_spatialdata(self, data, background="black",
                                          centroid_indices=[], colors=None, cmap='jet',
                                          rotate=0, min_r=0.6, set_alpha=False, z=None,
                                          image_name="cellman_celltype_map"):
        """Back-compat shim: ``plot_celltypes_map(save_to_spatialdata=True)`` routes
        here → :meth:`write_celltype_map` (labels + RGB image + annotating table)."""
        try:
            self.write_celltype_map(colors=colors, cmap=cmap, background=background,
                                    z=z, min_r=min_r, set_alpha=set_alpha)
        except Exception as e:  # noqa: BLE001
            print(f"Warning: Failed to save celltype map to SpatialData: {e}")

    def plot_image(self, source, image_key='he', aspect='equal', **kwargs):
        """
        Display an H&E / morphology / staining image in global µm coordinates.

        Call this before :meth:`plot_downsampled_vector` in the same code
        cell so the two plots overlay on the same axes (matplotlib uses
        the same y-flipped convention thanks to the inverted ``extent``).

        The function accepts two input forms:

        1. **File path** (``source`` is ``str`` / ``Path``) — opens an
           OME-TIFF / PNG / JPG with PIL. The image is assumed to span
           exactly the transcript bounding box stored in
           ``original_coordinate_bounds`` (typical post-crop H&E tile).
           Extent in global µm is computed as
           ``original_coordinate_bounds * (x_scale, y_scale)``.

        2. **SpatialData** (``source`` is a ``spatialdata.SpatialData``) —
           the image element selected by ``image_key`` is rendered using
           its own global extent (via ``spatialdata.get_extent``). This
           respects whatever transformation lives on the element, which
           is the correct alignment with cell/transcript coordinates.

        Parameters
        ----------
        source : str, pathlib.Path, or spatialdata.SpatialData
            Either a path to an image file, or a SpatialData object
            containing the desired image element.
        image_key : str, default ``'he'``
            Image element key when ``source`` is a SpatialData object.
            Use ``'morphology_focus'`` (Xenium), ``'DAPI'`` (Merscope),
            ``'hires_image'`` (Visium), etc., to render the morphology
            channel instead of H&E. Ignored when ``source`` is a path.
        aspect : str
            Aspect ratio passed to ``plt.imshow``. Default ``'equal'``.
        **kwargs
            Additional keyword arguments forwarded to ``plt.imshow``.
        """
        import matplotlib.pyplot as plt
        from pathlib import Path as _Path

        # ---------- helper: pick a 2D RGB or grayscale array for display ----------
        def _display_array(arr):
            arr = np.asarray(arr)
            # spatialdata stores images as (c, y, x); xarray DataArray .values
            # is the same. Move channel to last axis if so.
            if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (3, 4):
                arr = np.moveaxis(arr, 0, -1)
            if arr.ndim == 3:
                if arr.shape[-1] == 1:
                    return arr[..., 0]
                if arr.shape[-1] == 3:
                    return arr  # RGB
                if arr.shape[-1] == 4:
                    # 4-channel can be RGBA (uint8) or multi-channel morphology
                    # (e.g. Xenium morphology_focus, uint16). Heuristic by dtype.
                    if arr.dtype == np.uint8:
                        return arr[..., :3]
                    return arr[..., 0]   # use first channel (typically DAPI)
            return arr  # 2D grayscale

        # ---------- branch on input type ----------
        is_path = isinstance(source, (str, _Path))

        if is_path:
            from PIL import Image as _Image
            he_img = np.array(_Image.open(str(source)).convert('RGB'))

            x_scale = self.x_scale if self.x_scale is not None else 1.0
            y_scale = self.y_scale if self.y_scale is not None else 1.0

            extent = None
            if hasattr(self, 'original_coordinate_bounds'):
                x_bounds = self.original_coordinate_bounds[0]
                y_bounds = self.original_coordinate_bounds[1]
                x_min = x_bounds[0] * x_scale
                x_max = x_bounds[1] * x_scale
                y_min = y_bounds[0] * y_scale
                y_max = y_bounds[1] * y_scale
                # extent = [left, right, bottom, top]; rows top→bottom = y_min→y_max
                extent = [x_min, x_max, y_max, y_min]
        else:
            try:
                import spatialdata as _sd
            except ImportError as e:
                raise ImportError(
                    "plot_he with a SpatialData input requires the spatialdata "
                    "package."
                ) from e
            try:
                import spatialdata_plot  # noqa: F401  (registers .pl accessor)
            except ImportError as e:
                raise ImportError(
                    "plot_he with a SpatialData input requires the "
                    "spatialdata-plot package."
                ) from e
            if not isinstance(source, _sd.SpatialData):
                raise TypeError(
                    f"plot_he source must be a path or SpatialData, "
                    f"got {type(source).__name__}."
                )
            if image_key not in source.images:
                raise KeyError(
                    f"sdata.images has no key '{image_key}'. "
                    f"Available: {list(source.images.keys())}"
                )

            # ---- delegate rendering to spatialdata-plot ----
            # spatialdata-plot handles transformations correctly (rotation,
            # scaling, etc.) so the rendered raster is axis-aligned in
            # the requested coordinate system. plot_downsampled_vector can
            # then scatter onto the same axes for overlay.
            title              = kwargs.pop('title', image_key)
            coordinate_systems = kwargs.pop('coordinate_systems', 'global')
            figsize            = kwargs.pop('figsize', None)
            ax                 = kwargs.pop('ax', None)

            # Split render-side kwargs from show-side kwargs
            _render_keys = {'channel', 'cmap', 'norm', 'na_color',
                            'palette', 'alpha', 'scale'}
            render_kwargs = {k: kwargs.pop(k) for k in list(kwargs)
                             if k in _render_keys}

            if ax is None:
                if figsize is not None:
                    fig = plt.figure(figsize=figsize)
                    ax = fig.add_subplot(111)
                else:
                    ax = plt.gca()

            source.pl.render_images(image_key, **render_kwargs).pl.show(
                title=title,
                coordinate_systems=coordinate_systems,
                ax=ax,
                **kwargs,
            )
            ax.set_aspect(aspect)
            # spatialdata-plot renders the image onto ``ax`` but also appends a
            # colorbar axes and leaves *that* as the current axes. A subsequent
            # ``plot_downsampled_vector`` (plt.scatter) would then land on the
            # colorbar instead of the image and appear to vanish. Restore the
            # image axes as current so the overlay scatters onto the image.
            plt.sca(ax)
            return

        plt.imshow(he_img, extent=extent, aspect=aspect, **kwargs)

    # Backward-compat alias: the method was renamed plot_he → plot_image
    # (it renders any morphology/staining image, not just H&E).
    plot_he = plot_image

    def plot_downsampled_vector(self, c='red', s=0.3, alpha=0.45, **kwargs):
        """
        Scatter plot the downsampled vectors (H&E-guided local maxima) in µm
        coordinates.

        Call ``plot_he()`` first in the same code cell to overlay the vectors
        on the H&E image.

        Requires ``downsample_vector()`` to have been called first.

        Parameters
        ----------
        c : color
            Colour of the scatter dots. Default ``'red'``.
        s : float
            Size of the scatter dots. Default ``0.3``.
        alpha : float
            Transparency of the scatter dots. Default ``0.45``.
        **kwargs
            Additional keyword arguments forwarded to ``plt.scatter``.
        """
        import matplotlib.pyplot as plt

        if self.downsampled_vectors is None:
            raise ValueError('downsampled_vectors not set. Run downsample_vector() first.')

        plot_coords = self.downsampled_vectors_original_coords
        if plot_coords is None:
            raise ValueError('downsampled_vectors not set. Run downsample_vector() first.')

        x_coords, y_coords = plot_coords[0], plot_coords[1]

        # Map sampled coords (element/raw frame) into the global frame the
        # morphology image is drawn in. Prefer the full element→global affine
        # (scale + translation + rotation) captured by extract_spatialdata_scale;
        # fall back to the diagonal scale (legacy behaviour) when it is absent.
        M = getattr(self, '_global_affine', None)
        if M is not None:
            px = M[0, 0] * x_coords + M[0, 1] * y_coords + M[0, 2]
            py = M[1, 0] * x_coords + M[1, 1] * y_coords + M[1, 2]
        else:
            x_scale = self.x_scale if self.x_scale is not None else 1.0
            y_scale = self.y_scale if self.y_scale is not None else 1.0
            px = x_coords * x_scale
            py = y_coords * y_scale

        plt.scatter(px, py, s=s, c=c, alpha=alpha, **kwargs)