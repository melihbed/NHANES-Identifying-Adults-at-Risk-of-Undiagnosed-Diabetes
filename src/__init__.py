"""Reusable, tested implementation of the undiagnosed-diabetes screening pipeline.

The Jupyter notebooks in ``notebooks/`` are the exploratory narrative; this package
is the packaged version of the same logic so it can be unit-tested and re-run with a
single command (``python -m src.pipeline``).

Submodules: :mod:`src.config`, :mod:`src.data`, :mod:`src.features`,
:mod:`src.modeling`, :mod:`src.pipeline`.
"""

__all__ = ["config", "data", "features", "modeling", "pipeline"]
