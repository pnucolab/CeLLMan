"""CeLLMan — AI agentic workflow for automatic cell-type annotation
based on a knowledge graph of genes.

Subpackages
-----------
spatial
    Segmentation-free, multi-platform spatial cell-type mapping workflow
    (``import cellman.spatial as cms``). SSAM-derived KDE vector field.

The agentic annotation step (LLM + ontology server) ships as a skill bundle
under ``Agentic-Annotation/`` and is driven by your own LLM host, not imported
from here.
"""

__version__ = "0.1.0"
