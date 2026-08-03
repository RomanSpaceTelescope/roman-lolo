"""
Roman WFI Lamp-On/Lamp-Off CRNL Simulation Tool
=================================================
Simulates astronomical scenes with count-rate non-linearity (CRNL)
for characterizing the lamp-on/lamp-off flat-field method.

Author: Maxime Rizzo
Date: 2026-07-20

Code developed with assistance of AI tools (Anthropic/Claude)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
from astropy.table import Table
from astropy.modeling.models import Gaussian2D, Sersic2D
from astropy.convolution import Gaussian2DKernel, convolve
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# Roman WFI Instrument Parameters
# =============================================================================
@dataclass
class RomanWFIParams:
    """Roman Wide Field Instrument parameters."""
    nx: int = 4096                    # Pixels in x
    ny: int = 4096                    # Pixels in y
    pixel_scale: float = 0.11         # arcsec/pixel
    read_noise: float = 12.0          # e- rms (single read)
    dark_current: float = 0.015       # e-/s/pixel
    full_well: float = 200_000_000.0    # e- per pixel (increased for high pedestal tests)
    gain: float = 1.0                 # e-/DN
    psf_fwhm_arcsec: float = 0.15     # Approximate PSF FWHM in arcsec
    exposure_time: float = 140.0      # seconds (typical WFI exposure)

    @property
    def psf_fwhm_pixels(self) -> float:
        return self.psf_fwhm_arcsec / self.pixel_scale

    @property
    def psf_sigma_pixels(self) -> float:
        return self.psf_fwhm_pixels / 2.355


# =============================================================================
# CRNL Model
# =============================================================================
@dataclass
class CRNLModel:
    """
    Count-Rate Non-Linearity model as a power law.

    The measured count rate R_meas relates to the true rate s_true as:
        R_meas = s_true * (s_true / s_cal)^(alpha - 1)

    Or equivalently:
        R_meas = s_cal * (s_true / s_cal)^alpha

    Where alpha is very close to 1.0 (e.g., 1.01 means ~1% CRNL per decade).
    """
    alpha: float = 1.005              # Power-law exponent (very close to 1)
    s_cal: float = 1000.0             # Reference rate (e-/s) where CRNL = 0

    def apply(self, true_rate: np.ndarray) -> np.ndarray:
        """Apply CRNL to convert true rate to measured rate."""
        # Avoid division by zero
        safe_rate = np.maximum(true_rate, 1e-10)
        measured = self.s_cal * (safe_rate / self.s_cal) ** self.alpha
        return measured

    def inverse(self, measured_rate: np.ndarray) -> np.ndarray:
        """Invert CRNL to recover true rate from measured rate."""
        safe_rate = np.maximum(measured_rate, 1e-10)
        true_rate = self.s_cal * (safe_rate / self.s_cal) ** (1.0 / self.alpha)
        return true_rate

    def fractional_error(self, true_rate: np.ndarray) -> np.ndarray:
        """Compute fractional error (R_meas - s_true) / s_true."""
        measured = self.apply(true_rate)
        return (measured - true_rate) / true_rate


# =============================================================================
# Scene Generator
# =============================================================================
@dataclass
class SceneGenerator:
    """
    Generates simulated astronomical scenes for Roman WFI.

    Can create:
    - Point sources (stars) from a catalog or randomly placed
    - Extended sources (galaxies) with Sersic profiles
    - Uniform sky background
    - Flat-field lamp pedestals
    """
    wfi: RomanWFIParams = field(default_factory=RomanWFIParams)
    rng_seed: int = 42

    def __post_init__(self):
        self.rng = np.random.default_rng(self.rng_seed)

    def generate_star_catalog(self, n_stars: int = 500,
                              flux_min: float = 3,
                              flux_max: float = 50000.0,
                              x_max: Optional[float] = None,
                              y_max: Optional[float] = None) -> Table:
        """
        Generate a random star catalog.

        Parameters
        ----------
        n_stars : int
            Number of stars to place
        flux_min, flux_max : float
            Min/max flux in e-/s (power-law distributed)
        x_max, y_max : float or None
            Image bounds. If None, uses full WFI size.

        Returns
        -------
        catalog : astropy Table
            Columns: x_pix, y_pix, flux_es (e-/s)
        """
        if x_max is None:
            x_max = self.wfi.nx
        if y_max is None:
            y_max = self.wfi.ny

        margin = 5
        x = self.rng.uniform(margin, x_max - margin, n_stars).astype(float)
        y = self.rng.uniform(margin, y_max - margin, n_stars).astype(float)

        # Power-law flux distribution (more faint stars)
        # I realize this is an approximation, but it's not too important right now
        u = self.rng.uniform(0, 1, n_stars).astype(float)
        flux = flux_min * (flux_max / flux_min) ** u

        catalog = Table()
        catalog['x_pix'] = x
        catalog['y_pix'] = y
        catalog['flux_es'] = flux  # e-/s total flux
        catalog['type'] = 'star'

        return catalog

    def generate_galaxy_catalog(self, n_galaxies: int = 100,
                                flux_min: float = 1.0,
                                flux_max: float = 200.0,
                                x_max: Optional[float] = None,
                                y_max: Optional[float] = None) -> Table:
        """
        Generate a random galaxy catalog.

        Parameters
        ----------
        n_galaxies : int
            Number of galaxies
        flux_min, flux_max : float
            Min/max flux in e-/s
        x_max, y_max : float or None
            Image bounds. If None, uses full WFI size.

        Returns
        -------
        catalog : astropy Table
        """
        if x_max is None:
            x_max = self.wfi.nx
        if y_max is None:
            y_max = self.wfi.ny

        margin = 5
        x = self.rng.uniform(margin, x_max - margin, n_galaxies)
        y = self.rng.uniform(margin, y_max - margin, n_galaxies)

        u = self.rng.uniform(0, 1, n_galaxies)
        flux = flux_min * (flux_max / flux_min) ** u

        # Galaxy sizes (half-light radius in pixels)
        r_eff = self.rng.lognormal(mean=np.log(3.0), sigma=0.5, size=n_galaxies)
        # Sersic indices (1=exponential disk, 4=de Vaucouleurs)
        sersic_n = self.rng.uniform(0.5, 4.0, n_galaxies)
        # Ellipticities
        ellip = self.rng.uniform(0.0, 0.7, n_galaxies)
        # Position angles
        theta = self.rng.uniform(0, np.pi, n_galaxies)

        catalog = Table()
        catalog['x_pix'] = x
        catalog['y_pix'] = y
        catalog['flux_es'] = flux
        catalog['r_eff'] = r_eff
        catalog['sersic_n'] = sersic_n
        catalog['ellip'] = ellip
        catalog['theta'] = theta
        catalog['type'] = 'galaxy'

        return catalog

    def render_scene(self, star_catalog: Optional[Table] = None,
                     galaxy_catalog: Optional[Table] = None,
                     sky_background: float = 0.5,
                     stamp_size: int = 51,
                     image_size: Optional[int] = None) -> np.ndarray:
        """
        Render the true count-rate image (e-/s/pixel) from catalogs.

        Parameters
        ----------
        star_catalog : Table or None
            Star catalog (generated if None)
        galaxy_catalog : Table or None
            Galaxy catalog (generated if None)
        sky_background : float
            Sky background in e-/s/pixel
        stamp_size : int
            Postage stamp size for rendering objects
        image_size : int or None
            Output image size (square). If None, uses full WFI size (4096×4096).
            Plate scale remains unchanged.

        Returns
        -------
        scene : ndarray
            True count rate in e-/s/pixel
        """
        if image_size is not None:
            nx, ny = image_size, image_size
        else:
            nx, ny = self.wfi.nx, self.wfi.ny

        # Start with sky background
        scene = np.full((ny, nx), sky_background, dtype=np.float64)

        # Add dark current
        scene += self.wfi.dark_current

        # Generate catalogs if not provided
        if star_catalog is None:
            n_stars = int(500 * (nx * ny) / (4096 * 4096))
            star_catalog = self.generate_star_catalog(n_stars=max(10, n_stars),
                                                      x_max=nx, y_max=ny)

        if galaxy_catalog is None:
            n_gal = int(100 * (nx * ny) / (4096 * 4096))
            galaxy_catalog = self.generate_galaxy_catalog(n_galaxies=max(5, n_gal),
                                                          x_max=nx, y_max=ny)

        # Render stars as PSF (Gaussian approximation)
        sigma = self.wfi.psf_sigma_pixels
        half = stamp_size // 2

        print(f"  Rendering {len(star_catalog)} stars...")
        for row in star_catalog:
            cx = row['x_pix']
            cy = row['y_pix']
            ix, iy = int(cx), int(cy)

            # Bounds check
            x0 = max(0, ix - half)
            x1 = min(nx, ix + half + 1)
            y0 = max(0, iy - half)
            y1 = min(ny, iy + half + 1)

            if x1 <= x0 or y1 <= y0:
                continue

            # Create Gaussian PSF stamp
            yy, xx = np.mgrid[y0:y1, x0:x1]
            psf = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
            psf /= psf.sum()  # Normalize

            scene[y0:y1, x0:x1] += row['flux_es'] * psf

        # Render galaxies as Sersic profiles convolved with PSF
        print(f"  Rendering {len(galaxy_catalog)} galaxies...")
        for row in galaxy_catalog:
            cx = row['x_pix']
            cy = row['y_pix']
            ix, iy = int(cx), int(cy)
            r_eff = row['r_eff']

            # Use larger stamp for galaxies
            gal_half = int(max(half, 5 * r_eff))
            x0 = max(0, ix - gal_half)
            x1 = min(nx, ix + gal_half + 1)
            y0 = max(0, iy - gal_half)
            y1 = min(ny, iy + gal_half + 1)

            if x1 <= x0 or y1 <= y0:
                continue

            yy, xx = np.mgrid[y0:y1, x0:x1]

            # Simplified Sersic: use exponential (n=1) for speed
            # with ellipticity
            # I admit an AI wrote this code
            dx = xx - cx
            dy = yy - cy
            cos_t = np.cos(row['theta'])
            sin_t = np.sin(row['theta'])
            dx_rot = dx * cos_t + dy * sin_t
            dy_rot = -dx * sin_t + dy * cos_t

            q = 1.0 - row['ellip']
            r = np.sqrt(dx_rot**2 + (dy_rot / max(q, 0.1))**2)

            # Sersic profile (approximate)
            bn = 1.9992 * row['sersic_n'] - 0.3271
            profile = np.exp(-bn * ((r / max(r_eff, 0.5))**(1.0 / row['sersic_n']) - 1))
            profile /= profile.sum() + 1e-30

            scene[y0:y1, x0:x1] += row['flux_es'] * profile

        return scene


# =============================================================================
# Lamp-On / Lamp-Off Simulator
# =============================================================================
@dataclass
class LampOnOffSimulator:
    """
    Simulates the lamp-on/lamp-off method for CRNL characterization.

    The method works by:
    1. Observe sky scene ("lamp off") -> measure R_off
    2. Observe sky scene + lamp pedestal ("lamp on") -> measure R_on
    3. Lamp signal = R_on - R_off (affected by CRNL)
    """
    wfi: RomanWFIParams = field(default_factory=RomanWFIParams)
    crnl: CRNLModel = field(default_factory=CRNLModel)
    rng_seed: int = 42

    def __post_init__(self):
        self.rng = np.random.default_rng(self.rng_seed)

    def add_lamp_pedestal(self, scene: np.ndarray,
                          pedestal_rate: float) -> np.ndarray:
        """
        Add a uniform lamp pedestal to the scene.

        Parameters
        ----------
        scene : ndarray
            True count rate image (e-/s/pixel)
        pedestal_rate : float
            Lamp pedestal rate in e-/s/pixel

        Returns
        -------
        scene_with_lamp : ndarray
            Scene + lamp pedestal (true rates)
        """
        return scene + pedestal_rate

    def observe(self, true_rate: np.ndarray,
                exposure_time: Optional[float] = None,
                apply_crnl: bool = True,
                add_noise: bool = True) -> np.ndarray:
        """
        Simulate an observation: apply CRNL, integrate, add noise.

        Parameters
        ----------
        true_rate : ndarray
            True count rate in e-/s/pixel
        exposure_time : float or None
            Integration time (uses default if None)
        apply_crnl : bool
            Whether to apply CRNL
        add_noise : bool
            Whether to add photon + read noise

        Returns
        -------
        measured_rate : ndarray
            Measured count rate in e-/s/pixel (after CRNL)
        """
        if exposure_time is None:
            exposure_time = self.wfi.exposure_time

        # Apply CRNL to the rate
        if apply_crnl:
            measured_rate = self.crnl.apply(true_rate)
        else:
            measured_rate = true_rate.copy()

        # Add noise (Poisson + read noise) if requested
        if add_noise:
            # Total electrons accumulated
            total_e = measured_rate * exposure_time

            # Poisson noise on accumulated signal
            noisy_e = self.rng.poisson(np.maximum(total_e, 0).astype(np.float64))

            # Add read noise
            noisy_e = noisy_e + self.rng.normal(0, self.wfi.read_noise,
                                                 size=true_rate.shape)

            # Convert back to rate
            measured_rate = noisy_e / exposure_time

        # Clip at full well
        # NOTE: this has caused me some issues and is fake for the moment
        # the full well is artifically large just for convenience
        max_rate = self.wfi.full_well / exposure_time
        measured_rate = np.clip(measured_rate, 0, max_rate)

        return measured_rate

    def lamp_on_off_experiment(self, scene: np.ndarray,
                               pedestal_rates: List[float],
                               n_repeats: int = 1,
                               apply_crnl: bool = True,
                               add_noise: bool = True) -> dict:
        """
        Run a full lamp-on/lamp-off experiment at multiple pedestal levels and dithers.

        Parameters
        ----------
        scene : ndarray
            True sky scene rate (e-/s/pixel)
        pedestal_rates : list of float
            Lamp pedestal rates to test (e-/s/pixel)
        n_repeats : int
            Number of repeat observations per pedestal level
        apply_crnl : bool
            Whether to apply CRNL
        add_noise : bool
            Whether to add noise

        Returns
        -------
        results : dict
            Contains measured rates, differences, and CRNL diagnostics
        """
        results = {
            'pedestal_rates': np.array(pedestal_rates),
            'lamp_off': [],
            'lamp_on': [],
            'measured_lamp_signal': [],
            'true_lamp_signal': np.array(pedestal_rates),
            'scene_true_rate': scene,
        }

        # Lamp-off observation(s)
        print("Observing lamp-off frames...")
        lamp_off_stack = []
        for i in range(n_repeats):
            obs = self.observe(scene, apply_crnl=apply_crnl, add_noise=add_noise)
            lamp_off_stack.append(obs)
        lamp_off_mean = np.mean(lamp_off_stack, axis=0)
        results['lamp_off'] = lamp_off_mean

        # Lamp-on observations at each pedestal level
        for ped_rate in pedestal_rates:
            print(f"Observing lamp-on at pedestal = {ped_rate:.1f} e-/s...")
            scene_with_lamp = self.add_lamp_pedestal(scene, ped_rate)

            lamp_on_stack = []
            for i in range(n_repeats):
                obs = self.observe(scene_with_lamp, apply_crnl=apply_crnl,
                                   add_noise=add_noise)
                lamp_on_stack.append(obs)
            lamp_on_mean = np.mean(lamp_on_stack, axis=0)

            results['lamp_on'].append(lamp_on_mean)

            # Measured lamp signal = lamp_on - lamp_off
            diff = lamp_on_mean - lamp_off_mean
            results['measured_lamp_signal'].append(diff)

        results['lamp_on'] = results['lamp_on']
        results['measured_lamp_signal'] = results['measured_lamp_signal']

        return results

    def dithered_lamp_on_off_experiment(
            self,
            scene: np.ndarray,
            dither_offsets_arcsec: List[Tuple[float, float]],
            pedestal_rates: List[float],
            apply_crnl: bool = True,
            add_noise: bool = True) -> dict:
        """
        Run a dithered lamp-on/lamp-off experiment.

        At each dither position the scene is shifted on the detector while the
        lamp pedestal remains spatially uniform.  One lamp-off and one lamp-on
        frame (per pedestal level) is recorded at every pointing.

        Parameters
        ----------
        scene : ndarray
            True sky scene rate (e-/s/pixel)
        dither_offsets_arcsec : list of (dx_arcsec, dy_arcsec)
            Sky offsets for each dither pointing; positive x = east, positive y = north.
        pedestal_rates : list of float
            Lamp pedestal rates to test (e-/s/pixel)
        apply_crnl : bool
        add_noise : bool

        Returns
        -------
        results : dict
            'dither_offsets_arcsec'  : (N_d, 2) array of sky offsets
            'dither_offsets_pix'     : (N_d, 2) array of pixel shifts
            'pedestal_rates'         : (N_p,) array
            'scene_true'             : reference scene image
            'lamp_off'               : list[N_d] of lamp-off images
            'lamp_on'                : list[N_p] of list[N_d] of lamp-on images
            'lamp_signal'            : list[N_p] of list[N_d] of (on - off) images
        """
        pixel_scale = self.wfi.pixel_scale
        dither_offsets_pix = [
            (dx / pixel_scale, dy / pixel_scale)
            for dx, dy in dither_offsets_arcsec
        ]
        n_dithers = len(dither_offsets_arcsec)
        n_peds = len(pedestal_rates)

        print(f"Dithered LOLO: {n_dithers} dithers × {n_peds} pedestal(s)...")

        lamp_off_list = []
        lamp_on_by_ped = [[] for _ in range(n_peds)]
        lamp_signal_by_ped = [[] for _ in range(n_peds)]

        for d, (dx_pix, dy_pix) in enumerate(dither_offsets_pix):
            dx_int = int(round(dx_pix))
            dy_int = int(round(dy_pix))
            print(f"  Dither {d+1}/{n_dithers}: "
                  f"({dither_offsets_arcsec[d][0]:+.2f}\", "
                  f"{dither_offsets_arcsec[d][1]:+.2f}\") = "
                  f"({dx_int:+d}, {dy_int:+d}) pix")

            # Shift scene by integer pixels (roll wraps edges; fine for large images)
            shifted = np.roll(np.roll(scene, dy_int, axis=0), dx_int, axis=1)

            obs_off = self.observe(shifted, apply_crnl=apply_crnl, add_noise=add_noise)
            lamp_off_list.append(obs_off)

            for p, ped in enumerate(pedestal_rates):
                obs_on = self.observe(shifted + ped,
                                      apply_crnl=apply_crnl, add_noise=add_noise)
                lamp_on_by_ped[p].append(obs_on)
                lamp_signal_by_ped[p].append(obs_on - obs_off)

        return {
            'dither_offsets_arcsec': np.array(dither_offsets_arcsec),
            'dither_offsets_pix':    np.array(dither_offsets_pix),
            'pedestal_rates':        np.array(pedestal_rates),
            'scene_true':            scene,
            'lamp_off':              lamp_off_list,
            'lamp_on':               lamp_on_by_ped,
            'lamp_signal':           lamp_signal_by_ped,
        }


# =============================================================================
# Visualization
# =============================================================================
def plot_scene(scene: np.ndarray, title: str = "Simulated Scene",
               vmin: Optional[float] = None, vmax: Optional[float] = None):
    """Plot a scene image."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    if vmin is None:
        vmin = np.percentile(scene, 1)
    if vmax is None:
        vmax = np.percentile(scene, 99.5)
    im = ax.imshow(scene, origin='lower', cmap='magma',
                   norm=LogNorm(vmin=max(vmin, 0.01), vmax=vmax))
    plt.colorbar(im, ax=ax, label='Count Rate (e⁻/s/pixel)')
    ax.set_title(title)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    plt.tight_layout()
    return fig


def plot_crnl_curve(crnl_model: CRNLModel,
                    rate_range: Tuple[float, float] = (0.1, 50000)):
    """Plot the CRNL power-law curve."""
    rates = np.logspace(np.log10(rate_range[0]), np.log10(rate_range[1]), 200)
    frac_error = crnl_model.fractional_error(rates) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rates, frac_error, 'b-', linewidth=2)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(crnl_model.s_cal, color='red', linestyle=':',
               label=f's_cal = {crnl_model.s_cal:.0f} e⁻/s')
    ax.set_xscale('log')
    ax.set_xlabel('True Count Rate (e⁻/s/pixel)')
    ax.set_ylabel('CRNL Fractional Error (%)')
    ax.set_title(f'CRNL Power Law: α = {crnl_model.alpha:.5f}\n'
                 f's_meas = s_cal × (s_true / s_cal)^α')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# =============================================================================
# FITS I/O
# =============================================================================
def save_dithered_to_fits(filename: str, dithered_results: dict,
                          crnl_model: CRNLModel, wfi: RomanWFIParams):
    """
    Save dithered lamp-on/lamp-off results to a multi-extension FITS file.

    Extension layout
    ----------------
    [0]  PRIMARY       : metadata only (no image data)
    [1]  SCENE         : true sky scene (e-/s/pixel)
    [d+2]  LOFF_D{d}  : lamp-off image at dither d
    [N_d+2 + p*N_d + d]  LON_P{p}D{d}  : lamp-on image at pedestal p, dither d
    [N_d+2 + N_p*N_d + p*N_d + d]  SIG_P{p}D{d}  : lamp signal (on - off)

    All images are float32, in units of e-/s/pixel.
    """
    hdul = fits.HDUList()

    pedestal_rates = dithered_results['pedestal_rates']
    offsets_arcsec = dithered_results['dither_offsets_arcsec']
    n_d = len(dithered_results['lamp_off'])
    n_p = len(pedestal_rates)

    # Primary HDU — metadata only
    phdr = fits.Header()
    phdr['TELESCOP'] = 'ROMAN'
    phdr['INSTRUME'] = 'WFI'
    phdr['BUNIT']    = 'e-/s'
    phdr['CRNLALPH'] = (crnl_model.alpha,  'CRNL power-law exponent alpha')
    phdr['CRNLRREF'] = (crnl_model.s_cal,  'CRNL reference rate (e-/s)')
    phdr['EXPTIME']  = (wfi.exposure_time,  'Exposure time (s)')
    phdr['PIXSCALE'] = (wfi.pixel_scale,    'Pixel scale (arcsec/pix)')
    phdr['RDNOISE']  = (wfi.read_noise,     'Read noise (e-)')
    phdr['DARKCUR']  = (wfi.dark_current,   'Dark current (e-/s/pix)')
    phdr['N_DITHER'] = (n_d,               'Number of dither positions')
    phdr['N_PED']    = (n_p,               'Number of pedestal levels')
    for d, (dx, dy) in enumerate(offsets_arcsec):
        phdr[f'DX_D{d:02d}'] = (float(dx), f'Dither {d} X offset (arcsec)')
        phdr[f'DY_D{d:02d}'] = (float(dy), f'Dither {d} Y offset (arcsec)')
    for p, ped in enumerate(pedestal_rates):
        phdr[f'PED_{p:02d}'] = (float(ped), f'Pedestal {p} rate (e-/s/pix)')
    hdul.append(fits.PrimaryHDU(header=phdr))

    # Scene
    shdr = fits.Header()
    shdr['EXTNAME'] = 'SCENE'
    shdr['BUNIT']   = 'e-/s'
    shdr['COMMENT'] = 'True sky scene rate'
    hdul.append(fits.ImageHDU(
        data=dithered_results['scene_true'].astype(np.float32), header=shdr))

    # Lamp-off images — one per dither
    for d in range(n_d):
        hdr = fits.Header()
        hdr['EXTNAME'] = f'LOFF_D{d:1d}'[:8]
        hdr['DITHER']  = (d, 'Dither index')
        hdr['DX_ARCS'] = (float(offsets_arcsec[d, 0]), 'X offset (arcsec)')
        hdr['DY_ARCS'] = (float(offsets_arcsec[d, 1]), 'Y offset (arcsec)')
        hdr['BUNIT']   = 'e-/s'
        hdul.append(fits.ImageHDU(
            data=dithered_results['lamp_off'][d].astype(np.float32), header=hdr))

    # Lamp-on images — one per (pedestal, dither)
    for p, ped in enumerate(pedestal_rates):
        for d in range(n_d):
            hdr = fits.Header()
            hdr['EXTNAME'] = f'LON_P{p:1d}D{d:1d}'[:8]
            hdr['PEDIDX']  = (p,           'Pedestal index')
            hdr['PEDRATE'] = (float(ped),  'Pedestal rate (e-/s/pix)')
            hdr['DITHER']  = (d,           'Dither index')
            hdr['DX_ARCS'] = (float(offsets_arcsec[d, 0]), 'X offset (arcsec)')
            hdr['DY_ARCS'] = (float(offsets_arcsec[d, 1]), 'Y offset (arcsec)')
            hdr['BUNIT']   = 'e-/s'
            hdul.append(fits.ImageHDU(
                data=dithered_results['lamp_on'][p][d].astype(np.float32),
                header=hdr))

    # Lamp signal (on - off) images — one per (pedestal, dither)
    for p, ped in enumerate(pedestal_rates):
        for d in range(n_d):
            hdr = fits.Header()
            hdr['EXTNAME'] = f'SIG_P{p:1d}D{d:1d}'[:8]
            hdr['PEDIDX']  = (p,           'Pedestal index')
            hdr['PEDRATE'] = (float(ped),  'Pedestal rate (e-/s/pix)')
            hdr['DITHER']  = (d,           'Dither index')
            hdr['DX_ARCS'] = (float(offsets_arcsec[d, 0]), 'X offset (arcsec)')
            hdr['DY_ARCS'] = (float(offsets_arcsec[d, 1]), 'Y offset (arcsec)')
            hdr['BUNIT']   = 'e-/s'
            hdul.append(fits.ImageHDU(
                data=dithered_results['lamp_signal'][p][d].astype(np.float32),
                header=hdr))

    hdul.writeto(filename, overwrite=True)
    n_ext = len(hdul) - 1  # exclude primary
    print(f"Saved dithered LOLO to {filename}  ({n_ext} image extensions, "
          f"{n_d} dithers × {n_p} pedestals)")


def save_to_fits(filename: str, scene: np.ndarray, results: dict,
                 crnl_model: CRNLModel, wfi: RomanWFIParams):
    """Save simulation results to a FITS file."""
    hdul = fits.HDUList()

    # Primary: scene
    hdr = fits.Header()
    hdr['INSTRUME'] = 'WFI'
    hdr['TELESCOP'] = 'ROMAN'
    hdr['BUNIT'] = 'e-/s'
    hdr['CRNNLALP'] = (crnl_model.alpha, 'CRNL power-law exponent')
    hdr['CRNNLREF'] = (crnl_model.s_cal, 'CRNL reference rate (e-/s)')
    hdr['EXPTIME'] = (wfi.exposure_time, 'Exposure time (s)')
    hdr['PIXSCALE'] = (wfi.pixel_scale, 'Pixel scale (arcsec/pix)')
    hdr['RDNOISE'] = (wfi.read_noise, 'Read noise (e-)')
    hdr['DARKCRNT'] = (wfi.dark_current, 'Dark current (e-/s/pix)')
    primary = fits.PrimaryHDU(data=scene, header=hdr)
    hdul.append(primary)

    # Extensions: lamp-off
    hdr2 = fits.Header()
    hdr2['EXTNAME'] = 'LAMP_OFF'
    hdr2['BUNIT'] = 'e-/s'
    hdul.append(fits.ImageHDU(data=results['lamp_off'], header=hdr2))

    # Extensions: lamp-on at each pedestal
    for i, ped in enumerate(results['pedestal_rates']):
        hdr3 = fits.Header()
        hdr3['EXTNAME'] = f'LAMP_ON_{i}'
        hdr3['PEDESTAL'] = (ped, 'Lamp pedestal rate (e-/s)')
        hdr3['BUNIT'] = 'e-/s'
        hdul.append(fits.ImageHDU(data=results['lamp_on'][i], header=hdr3))

    hdul.writeto(filename, overwrite=True)
    print(f"Saved simulation to {filename}")


def simulate_LOLO_dataset(image_size: int = 512, alpha: float = 1.005,
             s_cal: float = 100.0, sky_bg: float = 0.3,
             pedestal_rates: Optional[List[float]] = None,
             n_stars: int = 200, n_galaxies: int = 50,
             star_catalog: Optional[Table] = None,
             galaxy_catalog: Optional[Table] = None,
             save_fits: bool = True):
    """
    Run a complete lamp-on/lamp-off CRNL simulation demo.

    Parameters
    ----------
    image_size : int
        Size of simulated image (square, for speed)
    alpha : float
        CRNL power-law exponent
    s_cal : float
        CRNL reference rate (e-/s)
    sky_bg : float
        Sky background rate (e-/s/pixel)
    pedestal_rates : list of float or None
        Lamp pedestal levels to test. Default covers a useful range.
    n_stars : int
        Number of stars in the scene
    n_galaxies : int
        Number of galaxies in the scene
    star_catalog : Table or None
        Pre-built star catalog (auto-generated if None)
    galaxy_catalog : Table or None
        Pre-built galaxy catalog (auto-generated if None)
    save_fits : bool
        Whether to save results to a FITS file

    Returns
    -------
    results : dict
        Full simulation and analysis results
    """
    print("=" * 60)
    print("  Roman WFI Lamp-On/Lamp-Off CRNL Simulation")
    print("=" * 60)

    # Setup
    wfi = RomanWFIParams()
    crnl_model = CRNLModel(alpha=alpha, s_cal=s_cal)
    scene_gen = SceneGenerator(wfi=wfi)
    simulator = LampOnOffSimulator(wfi=wfi, crnl=crnl_model)

    if pedestal_rates is None:
        pedestal_rates = [50.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0]

    print(f"\n  Image size: {image_size} x {image_size} pixels")
    print(f"  CRNL alpha: {alpha}")
    print(f"  CRNL S_cal: {s_cal} e-/s")
    print(f"  Sky background: {sky_bg} e-/s/pixel")
    print(f"  Pedestal levels: {pedestal_rates} e-/s")
    print(f"  Stars: {n_stars}, Galaxies: {n_galaxies}")
    print()

    # Generate scene
    print("Step 1: Generating astronomical scene...")
    if star_catalog is None:
        star_catalog = scene_gen.generate_star_catalog(n_stars=n_stars)
    if galaxy_catalog is None:
        galaxy_catalog = scene_gen.generate_galaxy_catalog(n_galaxies=n_galaxies)

    scene = scene_gen.render_scene(
        star_catalog=star_catalog,
        galaxy_catalog=galaxy_catalog,
        sky_background=sky_bg,
        image_size=image_size
    )
    print(f"  Scene rate range: [{scene.min():.2f}, {scene.max():.1f}] e-/s/pixel")
    print(f"  Scene median: {np.median(scene):.3f} e-/s/pixel")
    print()

    # Plot the CRNL curve
    print("Step 2: Plotting CRNL model...")
    fig_crnl = plot_crnl_curve(crnl_model)
    plt.savefig('crnl_curve.png', dpi=150, bbox_inches='tight')
    print("  Saved: crnl_curve.png")

    # Plot the scene
    print("Step 3: Plotting simulated scene...")
    fig_scene = plot_scene(scene, title=f"Simulated Roman WFI Scene ({image_size}×{image_size})")
    plt.savefig('simulated_scene.png', dpi=150, bbox_inches='tight')
    print("  Saved: simulated_scene.png")

# =============================================================================
# Additional Visualization Functions
# =============================================================================
def plot_lamp_differences(results: dict, save_fits: bool = False, fits_prefix: str = 'lamp_diff'):
    """
    Plot lamp-on minus lamp-off difference images.

    Parameters
    ----------
    results : dict
        Results dictionary from lamp_on_off_experiment containing pedestal rates and lamp signals.
    save_fits : bool, optional
        If True, save each difference image to a FITS file (default: False).
    fits_prefix : str, optional
        Prefix for FITS filenames (default: 'lamp_diff'). Files will be named
        like '{fits_prefix}_ped{N:.0f}e.fits' for each pedestal level.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plotted differences.
    """
    n_peds = len(results['pedestal_rates'])
    ncols = min(3, n_peds)
    nrows = (n_peds + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    saved_files = []
    for i, ped in enumerate(results['pedestal_rates']):
        row_idx = i // ncols
        col_idx = i % ncols
        ax = axes[row_idx, col_idx]

        diff = results['measured_lamp_signal'][i]
        residual = (diff - ped) / ped * 100  # Percent deviation from expected

        im = ax.imshow(residual, origin='lower', cmap='RdBu_r',
                       vmin=-2, vmax=2)
        plt.colorbar(im, ax=ax, label='Deviation (%)')
        ax.set_title(f'Pedestal = {ped:.0f} e⁻/s\n'
                     f'Mean dev = {np.nanmean(residual):.3f}%')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')

        # Save to FITS if requested
        if save_fits:
            hdu = fits.PrimaryHDU(data=residual.astype(np.float32))
            hdu.header['TELESCOP'] = 'Roman'
            hdu.header['INSTRUME'] = 'WFI'
            hdu.header['PEDESTAL'] = (ped, 'Lamp pedestal level (e-/s/pixel)')
            hdu.header['BUNIT'] = 'percent'
            hdu.header['COMMENT'] = f'Lamp signal deviation from expected pedestal level'

            fits_filename = f'{fits_prefix}_ped{ped:.0f}e.fits'
            hdu.writeto(fits_filename, overwrite=True)
            saved_files.append(fits_filename)

    # Turn off unused axes
    for i in range(n_peds, nrows * ncols):
        row_idx = i // ncols
        col_idx = i % ncols
        axes[row_idx, col_idx].set_visible(False)

    plt.suptitle('Lamp Signal Deviation from Expected (CRNL Effect)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    # Print summary if FITS files were saved
    if save_fits and saved_files:
        print(f"Saved {len(saved_files)} difference images to FITS:")
        for fname in saved_files:
            print(f"  ✓ {fname}")

    return fig

# =============================================================================
# Utility: Generate Flat-Field Variations
# =============================================================================
def generate_flat_field(nx: int, ny: int,
                        low_freq_amplitude: float = 0.02,
                        high_freq_amplitude: float = 0.005,
                        dead_pixel_fraction: float = 0.001,
                        rng_seed: int = 99) -> np.ndarray:
    """
    Generate a realistic flat-field pattern.

    Parameters
    ----------
    nx, ny : int
        Image dimensions
    low_freq_amplitude : float
        Amplitude of large-scale QE variations
    high_freq_amplitude : float
        Amplitude of pixel-to-pixel variations
    dead_pixel_fraction : float
        Fraction of dead/hot pixels
    rng_seed : int
        Random seed

    Returns
    -------
    flat : ndarray
        Flat-field (multiplicative, centered on 1.0)
    """
    rng = np.random.default_rng(rng_seed)

    # Low-frequency variations (smooth)
    # Use a low-pass filtered random field
    raw = rng.normal(0, 1, (ny, nx))
    kernel = Gaussian2DKernel(x_stddev=nx // 8)
    smooth = convolve(raw, kernel, boundary='wrap')
    smooth = smooth / smooth.std() * low_freq_amplitude

    # High-frequency (pixel-to-pixel)
    pixel_var = rng.normal(0, high_freq_amplitude, (ny, nx))

    # Combine
    flat = 1.0 + smooth + pixel_var

    # Dead pixels
    n_dead = int(dead_pixel_fraction * nx * ny)
    dead_x = rng.integers(0, nx, n_dead)
    dead_y = rng.integers(0, ny, n_dead)
    flat[dead_y, dead_x] = 0.0

    # Hot pixels (high dark current)
    n_hot = n_dead // 5
    hot_x = rng.integers(0, nx, n_hot)
    hot_y = rng.integers(0, ny, n_hot)
    flat[hot_y, hot_x] = rng.uniform(1.5, 5.0, n_hot)

    return flat
