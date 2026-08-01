"""gwmg: modified-gravity constraints from GW standard sirens and LSS.

A Python 3 / CosmoSIS 3 pipeline implementing the Baker & Harrison (2020,
arXiv:2007.13791) framework. hi_class computes the Horndeski cosmology and the
modified GW luminosity distance; CosmoSIS combines the GW siren likelihood with
CMB, RSD and BAO data.

Public API:
    gwmg.gw_log_likelihood    GW siren log-likelihood
    gwmg.load_events          read a standard-siren data file
    gwmg.pipeline_dir()       path to the bundled configs/modules/data
"""
import os

from .likelihood import gw_log_likelihood, load_events  # noqa: F401

__version__ = "0.1.0"
__all__ = ["gw_log_likelihood", "load_events", "pipeline_dir", "__version__"]


def pipeline_dir():
    """Absolute path to the bundled pipeline (configs/, modules/, data/)."""
    return os.path.join(os.path.dirname(__file__), "pipeline")
