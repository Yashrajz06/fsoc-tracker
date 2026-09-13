"""Statistical validation of the noise and disturbance models.

Noise is tested statistically -- measured mean, variance and impulse density against what was
requested -- rather than by eyeball. A noise model that looks plausible but has the wrong
variance produces an SNR curve that is quietly wrong everywhere.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from src.config import AppConfig, load_config
from src.noise.atmospheric import AtmosphericParams, apply_atmosphere, transmission
from src.noise.disturbance import (
    JitterModel,
    JitterParams,
    PlatformMotion,
    PlatformMotionParams,
)
from src.noise.pipeline import KNOWN_STAGES, NoisePipeline
from src.noise.sensor import (
    MAX_LEVEL,
    GaussianNoiseParams,
    PoissonNoiseParams,
    SaltPepperParams,
    add_gaussian_noise,
    add_poisson_noise,
    add_salt_pepper,
    to_uint8,
)
from src.noise.turbulence import (
    BeamWander,
    BeamWanderParams,
    ScintillationModel,
    TurbulenceParams,
    gamma_gamma_parameters,
)

DT = 1.0 / 30.0


@pytest.fixture()
def rng() -> np.random.Generator:
    """A seeded generator, so every statistical assertion is reproducible."""
    return np.random.default_rng(20260910)


@pytest.fixture()
def flat() -> np.ndarray:
    """A large uniform mid-grey frame, giving tight statistics with headroom both ways."""
    return np.full((600, 600), 100, dtype=np.uint8)


# ------------------------------------------------------------------------------------------
# Sensor noise
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("sigma", [1.0, 5.0, 10.0, 20.0])
def test_gaussian_noise_has_the_requested_variance(flat, rng, sigma: float) -> None:
    """Measured standard deviation must match the requested sigma.

    The mean tolerance is derived from the standard error of the mean, ``sigma/sqrt(N)``, rather
    than being a fixed constant: at sigma=20 over 360k pixels the SEM is 0.033, so a fixed 0.1
    tolerance is only 3 SEM and fails by chance. Five SEM is a ~1-in-3.5-million false-failure
    rate, which is the right target for a test that runs on every commit.
    """
    out = add_gaussian_noise(flat, GaussianNoiseParams(sigma=sigma), rng)
    standard_error = sigma / math.sqrt(out.size)
    assert out.mean() == pytest.approx(100.0, abs=5.0 * standard_error)
    assert out.std() == pytest.approx(sigma, rel=0.03)


def test_gaussian_noise_is_not_clipped_by_the_stage(rng) -> None:
    """Clipping must be deferred to the end of the pipeline.

    Clipping inside the stage would bias the noise non-zero-mean wherever the signal sits near
    0 or 255, quietly corrupting exactly the statistics this file validates.
    """
    dark = np.zeros((400, 400), dtype=np.uint8)
    out = add_gaussian_noise(dark, GaussianNoiseParams(sigma=20.0), rng)
    assert out.min() < 0.0
    assert out.mean() == pytest.approx(0.0, abs=0.2)


def test_gaussian_noise_disabled_is_a_no_op(flat, rng) -> None:
    """A disabled stage must pass the frame through untouched."""
    out = add_gaussian_noise(flat, GaussianNoiseParams(enabled=False, sigma=20.0), rng)
    assert np.array_equal(out, flat.astype(np.float32))


def test_poisson_noise_variance_equals_its_mean(flat, rng) -> None:
    """Shot noise is signal-dependent: variance equals the mean.

    This is what makes centroiding precision scale as 1/sqrt(N) in the shot-noise limit.
    """
    out = add_poisson_noise(flat, PoissonNoiseParams(enabled=True, scale=1.0), rng)
    assert out.mean() == pytest.approx(100.0, rel=0.01)
    assert out.var() == pytest.approx(100.0, rel=0.05)


def test_poisson_noise_is_brighter_where_the_signal_is_brighter(rng) -> None:
    """Absolute noise must grow with signal level, unlike additive Gaussian noise."""
    dim = np.full((400, 400), 25, dtype=np.uint8)
    bright = np.full((400, 400), 200, dtype=np.uint8)
    params = PoissonNoiseParams(enabled=True, scale=1.0)
    assert add_poisson_noise(bright, params, rng).std() > add_poisson_noise(dim, params, rng).std()


def test_poisson_scale_controls_effective_snr(flat, rng) -> None:
    """A larger scale means more photons per grey level and therefore less relative noise."""
    noisy = add_poisson_noise(flat, PoissonNoiseParams(enabled=True, scale=0.25), rng)
    clean = add_poisson_noise(flat, PoissonNoiseParams(enabled=True, scale=4.0), rng)
    assert noisy.std() > clean.std()
    assert noisy.mean() == pytest.approx(clean.mean(), rel=0.02)


@pytest.mark.parametrize("density", [0.01, 0.05, 0.10])
def test_salt_pepper_density_is_exact(flat, rng, density: float) -> None:
    """Density must match exactly, not approximately.

    Pixels are chosen without replacement, so a scenario's stated noise level is what it gets.
    Independent per-pixel sampling would give a binomial spread around the target.
    """
    out = add_salt_pepper(flat, SaltPepperParams(enabled=True, density=density), rng)
    corrupted = np.count_nonzero((out == MAX_LEVEL) | (out == 0.0))
    assert corrupted / out.size == pytest.approx(density, abs=1e-6)


@pytest.mark.parametrize("salt_ratio", [0.0, 0.25, 0.5, 1.0])
def test_salt_pepper_ratio_is_respected(flat, rng, salt_ratio: float) -> None:
    """The split between salt and pepper must follow the configured ratio."""
    out = add_salt_pepper(flat, SaltPepperParams(enabled=True, density=0.10,
                                                 salt_ratio=salt_ratio), rng)
    salt = np.count_nonzero(out == MAX_LEVEL)
    pepper = np.count_nonzero(out == 0.0)
    assert salt / (salt + pepper) == pytest.approx(salt_ratio, abs=0.01)


def test_salt_pepper_leaves_other_pixels_untouched(flat, rng) -> None:
    """Only the selected fraction may change; impulse noise is not additive."""
    out = add_salt_pepper(flat, SaltPepperParams(enabled=True, density=0.05), rng)
    untouched = out[(out != MAX_LEVEL) & (out != 0.0)]
    assert np.all(untouched == 100.0)


def test_noise_stages_do_not_mutate_their_input(flat, rng) -> None:
    """Every stage is a pure function; a shared buffer must be safe to reuse."""
    original = flat.copy()
    add_gaussian_noise(flat, GaussianNoiseParams(sigma=20.0), rng)
    add_poisson_noise(flat, PoissonNoiseParams(enabled=True), rng)
    add_salt_pepper(flat, SaltPepperParams(enabled=True, density=0.1), rng)
    assert np.array_equal(flat, original)


def test_invalid_noise_parameters_are_rejected(flat, rng) -> None:
    """Nonsensical parameters must fail rather than silently produce wrong statistics."""
    with pytest.raises(ValueError, match="non-negative"):
        add_gaussian_noise(flat, GaussianNoiseParams(sigma=-1.0), rng)
    with pytest.raises(ValueError, match="must be positive"):
        add_poisson_noise(flat, PoissonNoiseParams(enabled=True, scale=0.0), rng)
    with pytest.raises(ValueError, match="density"):
        add_salt_pepper(flat, SaltPepperParams(enabled=True, density=1.5), rng)


def test_noise_is_reproducible_from_a_seed(flat) -> None:
    """Identical seeds must give identical noise, or report figures are not reproducible."""
    a = add_gaussian_noise(flat, GaussianNoiseParams(sigma=10.0), np.random.default_rng(7))
    b = add_gaussian_noise(flat, GaussianNoiseParams(sigma=10.0), np.random.default_rng(7))
    assert np.array_equal(a, b)


# ------------------------------------------------------------------------------------------
# Atmospheric degradation
# ------------------------------------------------------------------------------------------


def test_transmission_follows_beer_lambert() -> None:
    """``t = exp(-beta * d)``."""
    assert transmission(0.0) == pytest.approx(1.0)
    assert transmission(0.75, 1.0) == pytest.approx(math.exp(-0.75))
    with pytest.raises(ValueError, match="non-negative"):
        transmission(-0.1)


@pytest.mark.parametrize("preset", ["clear", "haze", "fog", "low_light"])
def test_contrast_is_reduced_by_exactly_the_transmission(preset: str, rng) -> None:
    """The Koschmieder blend scales scene *differences* by ``t``, independently of airlight.

    This is the property that makes an absolute intensity threshold fail: fog can leave a beacon
    lower in contrast while making the frame as a whole brighter.
    """
    presets = json.loads(open("config/default.json", encoding="utf-8").read())
    params = AtmosphericParams.from_mapping(
        presets["noise"]["atmospheric"]["presets"][preset])

    scene = np.zeros((200, 200), dtype=np.float32)
    scene[90:110, 90:110] = 200.0
    out = apply_atmosphere(scene, params, rng)

    expected = params.transmission * params.brightness_scale
    measured = (out.max() - np.median(out)) / 200.0
    assert measured == pytest.approx(expected, rel=0.02)


def test_fog_brightens_the_frame_while_reducing_contrast(rng) -> None:
    """High airlight raises mean level even as contrast falls -- the case that breaks fixed thresholds."""
    scene = np.zeros((200, 200), dtype=np.float32)
    scene[90:110, 90:110] = 200.0
    fog = AtmosphericParams(beta=0.75, airlight=180.0)
    out = apply_atmosphere(scene, fog, rng)
    assert out.mean() > scene.mean()
    assert (out.max() - out.min()) < (scene.max() - scene.min())


def test_clear_preset_is_a_near_no_op(rng) -> None:
    """With beta=0 and no airlight the scene must pass through unchanged."""
    scene = np.full((100, 100), 42.0, dtype=np.float32)
    out = apply_atmosphere(scene, AtmosphericParams(beta=0.0, airlight=0.0), rng)
    assert out == pytest.approx(scene)


def test_low_light_compounds_dimming_with_scattering(rng) -> None:
    """Brightness scale and transmission multiply."""
    scene = np.full((100, 100), 200.0, dtype=np.float32)
    params = AtmosphericParams(beta=0.10, airlight=0.0, brightness_scale=0.25)
    out = apply_atmosphere(scene, params, rng)
    assert out.mean() == pytest.approx(200.0 * 0.25 * math.exp(-0.10), rel=0.01)


def test_atmosphere_rejects_out_of_range_airlight(rng) -> None:
    """Airlight outside the 8-bit range is a configuration error."""
    with pytest.raises(ValueError, match="Airlight"):
        apply_atmosphere(np.zeros((4, 4), np.float32),
                         AtmosphericParams(airlight=300.0), rng)


# ------------------------------------------------------------------------------------------
# Turbulence
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("rytov", [0.1, 1.0, 4.0, 10.0])
def test_gamma_gamma_parameters_are_positive_and_ordered(rytov: float) -> None:
    """Large-scale alpha must exceed small-scale beta across the range."""
    alpha, beta = gamma_gamma_parameters(rytov)
    assert alpha > 0 and beta > 0
    assert alpha > beta


def test_gamma_gamma_parameters_match_the_documented_table() -> None:
    """Pin the values quoted in DESIGN section 4.3, which were corrected during Phase 2.

    An earlier draft quoted alpha 2.95 / beta 2.46 at sigma_R^2 = 1; that does not follow from
    the plane-wave relations at any parameterisation. Published tables differ because alpha and
    beta depend on wave model and aperture averaging, so the formula is the reference, not a
    remembered table.
    """
    assert gamma_gamma_parameters(0.1) == pytest.approx((21.59, 19.82), abs=0.02)
    assert gamma_gamma_parameters(1.0) == pytest.approx((4.39, 2.56), abs=0.02)
    assert gamma_gamma_parameters(10.0) == pytest.approx((5.69, 1.10), abs=0.02)


@pytest.mark.parametrize("model", ["lognormal", "gamma_gamma"])
@pytest.mark.parametrize("rytov", [0.1, 1.0])
def test_scintillation_is_unit_mean_with_the_predicted_index(model: str, rytov: float) -> None:
    """Turbulence redistributes intensity over time without changing average power.

    Unit mean keeps the Rytov variance the single knob controlling fade severity -- dimming is
    the atmospheric model's job. The scintillation index is asserted against its closed form,
    which is the model-independent severity measure.
    """
    scint = ScintillationModel(
        TurbulenceParams(enabled=True, model=model, rytov_variance=rytov),
        np.random.default_rng(11))
    samples = scint.sample_many(120_000)
    assert samples.mean() == pytest.approx(1.0, rel=0.02)
    measured_index = samples.var() / samples.mean() ** 2
    assert measured_index == pytest.approx(scint.scintillation_index, rel=0.05)


def test_stronger_turbulence_produces_deeper_fades() -> None:
    """Severity must increase with Rytov variance -- deep fades are what cause dropouts."""
    def fade_depth(rytov: float) -> float:
        scint = ScintillationModel(
            TurbulenceParams(enabled=True, model="gamma_gamma", rytov_variance=rytov),
            np.random.default_rng(5))
        return float(np.percentile(scint.sample_many(40_000), 1.0))

    assert fade_depth(4.0) < fade_depth(1.0) < fade_depth(0.1)


def test_scintillation_disabled_returns_unity() -> None:
    """A disabled model must not perturb intensity at all."""
    scint = ScintillationModel(TurbulenceParams(enabled=False), np.random.default_rng(0))
    assert all(scint.sample() == 1.0 for _ in range(50))


def test_unknown_scintillation_model_is_rejected() -> None:
    """Only implemented distributions are selectable."""
    with pytest.raises(ValueError, match="Unknown scintillation model"):
        ScintillationModel(TurbulenceParams(model="weibull"))


def test_beam_wander_is_mean_reverting_and_bounded() -> None:
    """The displacement must wander about zero, not drift away without bound."""
    params = BeamWanderParams(enabled=True, theta=0.5, sigma_px=5.0)
    wander = BeamWander(params, np.random.default_rng(2))
    offsets = np.array([wander.step(DT) for _ in range(20_000)])
    assert abs(offsets[:, 0].mean()) < 1.0
    assert offsets[:, 0].std() == pytest.approx(params.stationary_sigma_px, rel=0.25)


def test_beam_wander_stationary_sigma_matches_theory() -> None:
    """The long-run sigma of an OU process is ``sigma / sqrt(2*theta)``."""
    params = BeamWanderParams(enabled=True, theta=0.2, sigma_px=5.0)
    assert params.stationary_sigma_px == pytest.approx(5.0 / math.sqrt(0.4))


def test_beam_wander_disabled_stays_at_zero() -> None:
    """A disabled process must not move."""
    wander = BeamWander(BeamWanderParams(enabled=False), np.random.default_rng(0))
    for _ in range(100):
        assert wander.step(DT) == (0.0, 0.0)


# ------------------------------------------------------------------------------------------
# Mechanical disturbance
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("distribution", ["gaussian", "uniform"])
def test_jitter_never_exceeds_its_stated_maximum(distribution: str) -> None:
    """Specification parameter 23 states a maximum, so the Gaussian tail must be clipped."""
    model = JitterModel(JitterParams(max_px_per_frame=20.0, distribution=distribution),
                        np.random.default_rng(0))
    offsets = np.array([model.sample() for _ in range(20_000)])
    assert np.abs(offsets).max() <= 20.0


def test_jitter_is_zero_mean(rng) -> None:
    """Jitter is a zero-mean disturbance above the loop bandwidth, not a drift.

    That is what distinguishes it from platform motion and why the controller must not try to
    reject it: feeding it to a PID injects it rather than attenuating it.
    """
    model = JitterModel(JitterParams(max_px_per_frame=15.0), rng)
    offsets = np.array([model.sample() for _ in range(20_000)])
    assert abs(offsets[:, 0].mean()) < 0.2
    assert abs(offsets[:, 1].mean()) < 0.2


def test_jitter_sigma_property_matches_its_realised_spread(rng) -> None:
    """The advertised sigma must be the real one -- Phase 4 uses it as a floor for Kalman R."""
    params = JitterParams(max_px_per_frame=15.0, distribution="gaussian")
    offsets = np.array([JitterModel(params, rng).sample() for _ in range(20_000)])
    assert offsets[:, 0].std() == pytest.approx(params.sigma_px, rel=0.05)


def test_jitter_disabled_is_exactly_zero() -> None:
    """A disabled disturbance must not perturb the boresight at all."""
    model = JitterModel(JitterParams(enabled=False, max_px_per_frame=20.0))
    assert all(model.sample() == (0.0, 0.0) for _ in range(50))


@pytest.mark.parametrize("motion", ["linear", "circular", "random", "spiral", "figure8"])
def test_platform_motion_respects_the_per_frame_bound(motion: str) -> None:
    """Specification parameter 25 caps drift at 20 px/frame regardless of velocity settings."""
    params = PlatformMotionParams(enabled=True, type=motion, max_px_per_frame=5.0,
                                  velocity_x_px_s=1000.0, velocity_y_px_s=1000.0,
                                  amplitude_px=500.0, step_sigma_px=50.0)
    platform = PlatformMotion(params, np.random.default_rng(0))
    previous = (0.0, 0.0)
    for _ in range(600):
        current = platform.step(DT)
        assert math.dist(current, previous) <= 5.0 + 1e-9
        previous = current


def test_platform_motion_is_biased_not_zero_mean() -> None:
    """Linear drift must accumulate: this is the disturbance the integral term exists to reject."""
    platform = PlatformMotion(PlatformMotionParams(enabled=True, type="linear",
                                                   velocity_x_px_s=30.0,
                                                   velocity_y_px_s=15.0))
    offsets = np.array([platform.step(DT) for _ in range(300)])
    assert offsets[:, 0].mean() > 5.0  # a clear bias, unlike jitter
    assert offsets[-1, 0] > offsets[0, 0]


def test_platform_motion_disabled_stays_at_zero() -> None:
    """A disabled drift must not move the boresight."""
    platform = PlatformMotion(PlatformMotionParams(enabled=False, type="linear"))
    for _ in range(100):
        assert platform.step(DT) == (0.0, 0.0)


def test_unknown_platform_motion_type_is_rejected() -> None:
    """Only implemented drift types are selectable."""
    with pytest.raises(ValueError, match="Unknown platform motion type"):
        PlatformMotion(PlatformMotionParams(type="lissajous"))
