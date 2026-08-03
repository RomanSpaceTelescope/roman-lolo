"""Roman WFI CRNL simulation and photometry tools."""

from .roman_crnl_sim import (
    RomanWFIParams,
    CRNLModel,
    SceneGenerator,
    LampOnOffSimulator,
    plot_scene,
    plot_crnl_curve,
    save_to_fits,
    save_dithered_to_fits,
    simulate_LOLO_dataset,
    plot_lamp_differences,
    generate_flat_field,
)

from .romanphot import SourcePhotometry

__all__ = [
    'RomanWFIParams',
    'CRNLModel',
    'SceneGenerator',
    'LampOnOffSimulator',
    'SourcePhotometry',
    'plot_scene',
    'plot_crnl_curve',
    'save_to_fits',
    'save_dithered_to_fits',
    'simulate_LOLO_dataset',
    'plot_lamp_differences',
    'generate_flat_field',
]

__version__ = '0.1.0'
