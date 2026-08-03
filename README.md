# Roman LOLO - Lamp-On/Lamp-Off Simulation and Analysis

Tools for simulating count-rate non-linearity (CRNL) in the Roman Space Telescope's Wide Field Instrument (WFI) using lamp-on/lamp-off observations.

## Installation

Clone the repository and install in editable mode:

```bash
pip install -e .
```

Or with development dependencies:

```bash
pip install -e ".[dev]"
```

## Features

### CRNL Simulation (`roman_lolo.roman_crnl_sim`)

- **RomanWFIParams**: Instrument parameters for Roman WFI
- **CRNLModel**: Power-law count-rate non-linearity model
- **SceneGenerator**: Generate synthetic astronomical scenes with stars and galaxies
- **LampOnOffSimulator**: Simulate lamp-on/lamp-off observations with realistic noise
- Visualization and FITS I/O utilities

### Source Photometry (`roman_lolo.romanphot`)

- **SourcePhotometry**: Source detection and aperture/PSF photometry
- DS9 integration for interactive visualization
- Noise estimation and signal-to-noise calculations

## Quick Start

```python
from roman_lolo import (
    RomanWFIParams,
    CRNLModel,
    SceneGenerator,
    LampOnOffSimulator,
)

# Set up instrument and CRNL model
wfi = RomanWFIParams()
crnl = CRNLModel(alpha=1.005, s_cal=100.0)

# Generate a scene with stars and galaxies
scene_gen = SceneGenerator(wfi=wfi)
scene = scene_gen.render_scene(n_stars=200, n_galaxies=50)

# Simulate lamp-on/lamp-off observations
simulator = LampOnOffSimulator(wfi=wfi, crnl=crnl)
results = simulator.lamp_on_off_experiment(
    scene=scene,
    pedestal_rates=[50.0, 200.0, 500.0]
)
```

## Project Structure

```
roman-lolo/
├── src/
│   └── roman_lolo/
│       ├── __init__.py
│       ├── roman_crnl_sim.py      # Main CRNL simulation code
│       └── romanphot.py           # Photometry utilities
├── notebooks/
│   └── LOLO_clean.ipynb          # Example analysis notebook
├── setup.py
├── pyproject.toml
└── README.md
```

## Documentation

See example notebooks in `notebooks/` for detailed usage.

## Author

Maxime Rizzo

## License

MIT
