"""Emulator support: generate hi_class training sets for CosmoPower-style
emulators of the modified-gravity (Horndeski) sector."""
from .generate import (  # noqa: F401
    generate_training_set, generate_parallel, merge_training_sets,
    latin_hypercube, DEFAULT_RANGES, SMART_RANGES,
)
from .train import train_emulators, load_training_set, PARAM_NAMES  # noqa: F401
from .validate import (  # noqa: F401
    validate_emulators, fractional_errors, summarise_errors, write_report,
)
from .chi2 import (  # noqa: F401
    PlanckLiteTT, chi2_validation, write_chi2_report, parameter_bias,
)
from .predict import Emulator, Prediction, default_model_dir  # noqa: F401

__all__ = ["generate_training_set", "generate_parallel", "merge_training_sets",
           "latin_hypercube", "DEFAULT_RANGES", "SMART_RANGES",
           "train_emulators", "load_training_set", "PARAM_NAMES",
           "validate_emulators", "fractional_errors", "summarise_errors", "write_report",
           "PlanckLiteTT", "chi2_validation", "write_chi2_report", "parameter_bias",
           "Emulator", "Prediction", "default_model_dir"]
