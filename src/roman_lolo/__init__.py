"""Roman WFI CRNL simulation and photometry tools."""

from .romanphot import SourcePhotometry


def __getattr__(name):
    _crnl_sim_exports = {
        'RomanWFIParams', 'CRNLModel', 'SceneGenerator', 'LampOnOffSimulator',
        'plot_scene', 'plot_crnl_curve', 'save_to_fits', 'save_dithered_to_fits',
        'simulate_LOLO_dataset', 'plot_lamp_differences', 'generate_flat_field',
    }
    if name in _crnl_sim_exports:
        from . import roman_crnl_sim as _m
        return getattr(_m, name)
    raise AttributeError(f"module 'roman_lolo' has no attribute {name!r}")

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

__version__ = '0.3.0'
