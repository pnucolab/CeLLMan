"""Gene-panel selection for count-based spatial data: HVG ∪ Cell-Ontology markers.

The vector field for a count platform is built over a chosen gene panel. That panel
is the union of

  * data-driven **HVGs** (seurat_v3 on a coarse-bin subsample, minus a
    technical-gene denylist), and
  * **marker genes** from the Cell-Ontology knowledge-graph module for the sample's
    (species, organ), fetched via the read-only slim-index REST API.

The KG is reached ONLY through the REST slim-index endpoint (never raw neo4j bolt).
The server URL is **received as a parameter** — there is no hardcoded IP default:
resolution is ``server_url`` arg > ``CELLONTO_API`` env > ``None``. When it resolves
to ``None`` or the module is unreachable, the panel degrades to **HVG-only** (the
pipeline never dies on a missing KG).
"""

from __future__ import annotations

import os
import re
from typing import Iterable, List, Optional, Tuple

import numpy as np
import scanpy as sc
from anndata import AnnData

DEFAULT_HVG_N = 5000
DEFAULT_HVG_SUB = 100_000

# Technical / low-information gene denylists by species. Ribosomal, mitochondrial,
# hemoglobin, and (mouse) predicted-gene / Riken; (human) lncRNA / antisense /
# pseudogene naming patterns that add noise but no cell-type signal.
_DENY_MOUSE_PREFIX = ("mt-", "Rps", "Rpl", "Hba", "Hbb")
_DENY_HUMAN_PREFIX = ("MT-", "RPS", "RPL", "MRPL", "MRPS", "HBA", "HBB")
_DENY_HUMAN_RE = re.compile(
    r"^(ENSG|LINC|MIR|SNOR)|(-AS\d*$)|(-DT$)|(-IT\d*$)|(-OT\d*$)"
    r"|(^RP\d+-)|(^A[CLP]\d{6})|(^Z\d{5})|(\.\d+$)"
)


def _denier(species: str):
    """Return a predicate ``g -> bool`` marking technical genes to drop."""
    sp = (species or "").lower()
    if sp.startswith("human") or sp == "hs" or sp == "homo_sapiens":
        def _denied(g: str) -> bool:
            return (g.startswith(_DENY_HUMAN_PREFIX) or g == "MALAT1"
                    or bool(_DENY_HUMAN_RE.search(g)))
        return _denied
    # default: mouse
    def _denied(g: str) -> bool:
        return (g.startswith(_DENY_MOUSE_PREFIX) or bool(re.match(r"Gm\d", g))
                or g.endswith("Rik"))
    return _denied


def resolve_server(server_url: Optional[str] = None) -> Optional[str]:
    """Resolve the Cell-Ontology REST base URL by priority.

    ``server_url`` argument > ``CELLONTO_SERVER_URL`` env > ``CELLONTO_API`` env
    (back-compat) > ``None``. Returns ``None`` when none is set — callers then
    skip the KG (HVG-only). ``CELLONTO_SERVER_URL`` is the canonical name (same
    one ``Agentic-Annotation/scripts/helper.py`` reads); ``CELLONTO_API`` is
    kept for older setups. No server address is baked into source — pass the URL
    explicitly or set ``CELLONTO_SERVER_URL`` (e.g. ``https://cellman.pnucolab.com``).
    """
    return (server_url
            or os.environ.get("CELLONTO_SERVER_URL")
            or os.environ.get("CELLONTO_API")
            or None)


def _clean(g: str) -> str:
    """Normalize a marker symbol to match ``var_names`` (drop -/_ suffixes)."""
    return g.split("-")[0].split("_")[0]


def fetch_ontology_markers(species: str, organ: str,
                           server_url: Optional[str] = None,
                           slots: Iterable[str] = ("R", "S", "N", "DF"),
                           timeout: float = 60.0,
                           verbose: bool = True) -> set:
    """Fetch the marker-gene set for a module from the slim-index REST API.

    Collects genes from the requested term slots (Required / Supporting /
    Negative / Defining by default). Returns an empty set (with a warning) when
    the server is unresolved / unreachable / the module is missing — the caller
    then proceeds HVG-only.
    """
    url = resolve_server(server_url)
    if url is None:
        if verbose:
            print("[panel] no ontology server (arg/CELLONTO_API unset) → HVG-only")
        return set()
    try:
        import requests
        resp = requests.get(f"{url}/module/{species}/{organ}/slim-index",
                            timeout=timeout)
        resp.raise_for_status()
        onto = resp.json()
    except Exception as e:  # noqa: BLE001 — any failure ⇒ graceful HVG-only
        if verbose:
            print(f"[panel][WARN] ontology fetch failed → HVG-only: {e}")
        return set()

    markers = set()
    for t in onto.get("terms", {}).values():
        for slot in slots:
            for g in (t.get(slot) or []):
                markers.add(_clean(g))
    if verbose:
        print(f"[panel] slim-index {species}/{organ}: "
              f"n_terms={onto.get('n_terms', len(onto.get('terms', {})))} "
              f"version={onto.get('version')} → {len(markers)} marker genes")
    return markers


def select_hvg(adata: AnnData, n_top_genes: int = DEFAULT_HVG_N,
               n_sub: int = DEFAULT_HVG_SUB, species: str = "mouse",
               seed: int = 0, verbose: bool = True) -> set:
    """Top-``n_top_genes`` HVGs (seurat_v3 on a subsample), minus the denylist.

    ``adata`` should be a coarse-bin aggregation (denser bins give a cleaner
    variance estimate than fine/sparse bins). Falls back to all genes if the
    matrix has fewer than ``n_top_genes`` genes.
    """
    denied = _denier(species)
    if adata.n_vars <= n_top_genes:
        return {g for g in adata.var_names if not denied(g)}
    rng = np.random.default_rng(seed)
    n = min(n_sub, adata.n_obs)
    sub = adata[rng.choice(adata.n_obs, n, replace=False)].copy()
    sc.pp.highly_variable_genes(sub, n_top_genes=n_top_genes, flavor="seurat_v3")
    hvg = set(sub.var_names[sub.var["highly_variable"]].tolist())
    hvg = {g for g in hvg if not denied(g)}
    if verbose:
        print(f"[panel] HVG (seurat_v3 on {n} bins, denylist-filtered): {len(hvg)}")
    return hvg


def select_gene_panel(adata: AnnData, species: str, organ: str, *,
                      hvg: Optional[set] = None,
                      hvg_n: int = DEFAULT_HVG_N, n_hvg_sub: int = DEFAULT_HVG_SUB,
                      server_url: Optional[str] = None,
                      slots: Iterable[str] = ("R", "S", "N", "DF"),
                      seed: int = 0, verbose: bool = True
                      ) -> Tuple[List[str], List[str]]:
    """Build the gene panel = HVG ∪ (ontology markers present in ``adata``).

    Parameters
    ----------
    adata : AnnData
        Bins used for HVG selection (coarse-bin aggregation recommended). If
        ``hvg`` is given it is used verbatim and HVG is not recomputed here.
    species, organ : str
        Cell-Ontology module to fetch markers for.
    server_url : str, optional
        KG REST base URL. Priority arg > ``CELLONTO_API`` env > ``None``
        (HVG-only). Pass explicitly during development.

    Returns
    -------
    panel : list[str]
        Sorted union of HVG and in-data ontology markers.
    ontology_markers : list[str]
        Sorted ontology markers that are present in ``adata`` (subset of panel).
    """
    if hvg is None:
        hvg = select_hvg(adata, n_top_genes=hvg_n, n_sub=n_hvg_sub,
                         species=species, seed=seed, verbose=verbose)
    markers = fetch_ontology_markers(species, organ, server_url=server_url,
                                     slots=slots, verbose=verbose)
    full = set(map(str, adata.var_names))
    markers_in_data = markers & full
    panel = sorted((set(hvg) | markers_in_data) & full)
    if verbose:
        print(f"[panel] final panel: {len(panel)} genes "
              f"(HVG {len(hvg)} ∪ ontology-in-data {len(markers_in_data)})")
    return panel, sorted(markers_in_data)
