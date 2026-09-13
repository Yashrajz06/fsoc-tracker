"""Constant-velocity Kalman filter with adaptive measurement noise and validation gating.

Three uses, all essential (``docs/DESIGN.md`` section 6):

1. **Predictive feedforward** to the controller, cancelling the one-frame measurement lag that
   any feedback tracker following a moving reference otherwise carries.
2. **Validation gating** -- a Mahalanobis test that rejects the salt-and-pepper false positives
   the single-frame detector cannot eliminate on its own.
3. **Coasting** through dropouts, which is what makes the <=1 s re-acquisition budget reachable.

Adaptive R is the innovation point
----------------------------------
``R`` is derived per frame from the measured detection SNR rather than fixed, using the accuracy
law of section 5.3::

    sigma_meas ~ FWHM / (2 * SNR_aperture)

A *static* ``R`` mis-scales the validation gate exactly when it matters most. At low SNR the true
measurement scatter grows, but a fixed small ``R`` keeps the gate tight, so genuine detections
are rejected and lock is dropped -- directly attacking the <5% loss-rate requirement. At high SNR
the same fixed ``R`` is too loose and impulse false positives pass. Deriving ``R`` from a quantity
already computed costs nothing and makes gate and innovation covariance self-consistent across
the whole SNR range.

Bias sources are folded in as variance, deliberately
----------------------------------------------------
``clipped`` and ``saturated`` mark detections whose centroid carries a *known, bounded,
systematic* displacement rather than extra noise (Phase 1 and Phase 2 measured ~2 px and ~0.19 px
respectively). A Kalman filter has no representation for bias, so the honest engineering
treatment is to widen ``R`` by the measured bias budget::

    R = sigma_meas^2 + bias_budget^2

which makes the filter lean on its prediction in proportion to how displaced we know the
measurement to be. The magnitudes are traceable to the measurements that produced them, not
tuned. Note this is *not* the same as rejecting the measurement: a bounded bias is still
information, whereas an impulse outlier is not, and the gate handles that case separately.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

__all__ = ["KalmanParams", "GateResult", "KalmanFilterCV", "measurement_sigma"]

#: Chi-squared 99% critical value for 2 degrees of freedom, the default gate threshold.
CHI2_99_2DOF: float = 9.21



#: Empirical correction to the theoretical accuracy law, measured against the Phase 3 SNR sweep.
#:
#: ``sigma = FWHM / (2 * SNR)`` is an idealisation; our pipeline's measured per-axis error runs
#: above it. Comparing the law against measured error across three shapes and four SNR bands gave
#: a median ratio of **1.32** (spread 0.74-2.25 once the saturated high-SNR bins, where the law
#: breaks down entirely, are excluded). Using the uncorrected law would make ``R`` optimistic by
#: roughly a third, which tightens the validation gate and rejects genuine detections -- exactly
#: the failure mode adaptive ``R`` exists to prevent.
#:
#: The spread is wide enough that this is a calibration, not a constant of nature. The floor and
#: the bias budgets matter more than its precise value.
LAW_CALIBRATION: float = 1.32


def measurement_sigma(snr_aperture: Optional[float], fwhm_px: float,
                      floor_px: float = 0.02, ceiling_px: float = 20.0,
                      clipped: bool = False, saturated: bool = False,
                      clipped_bias_px: float = 2.0,
                      saturated_bias_px: float = 0.19,
                      calibration: float = LAW_CALIBRATION,
                      unobservable_sigma_px: float = 0.0,
                      last_known_sigma_px: Optional[float] = None) -> float:
    """Derive per-measurement position sigma from SNR and the known bias flags.

    Args:
        snr_aperture: Measured aperture SNR, or ``None`` when unavailable.
        fwhm_px: Spot scale in pixels.
        floor_px: Lower clamp. The pipeline cannot beat 8-bit quantisation, measured at
            0.007-0.013 px, so a floor near 0.02 px prevents an over-confident gate.
        ceiling_px: Upper clamp, so a near-zero SNR cannot produce an unbounded ``R``.
        clipped: Detection touches the frame edge.
        saturated: Detection core is saturated.
        clipped_bias_px: Measured worst-case inward bias from truncation (Phase 1).
        saturated_bias_px: Measured worst-case bias from a flattened peak (Phase 2).
        calibration: Empirical correction to the theoretical law. See :data:`LAW_CALIBRATION`.
        last_known_sigma_px: The most recent sigma derived from a *measured* SNR. Used when SNR
            is unmeasurable, so a transient measurement failure does not spike ``R``.
        unobservable_sigma_px: Standard deviation of disturbances the estimator cannot observe,
            added in quadrature. **Camera jitter belongs here.** The viewport is extracted at
            boresight *plus* jitter while the reported boresight angle excludes it, so converting
            a frame-local measurement into canvas coordinates carries the jitter as an error of
            up to the specified +-20 px/frame. That is unobservable by construction -- it is the
            same property that makes jitter uncontrollable -- so the filter must widen ``R`` to
            match rather than pretend to a precision it cannot have. Omitting it produced
            sigma ~0.03 px against innovations of 1-2 px, i.e. NIS in the thousands and every
            measurement gated out while detection was perfect.

    Returns:
        Effective position sigma in pixels, combining random error and bias budget in quadrature.

    Raises:
        ValueError: If ``fwhm_px`` is not positive.
    """
    if fwhm_px <= 0:
        raise ValueError(f"FWHM must be positive, got {fwhm_px}")

    if snr_aperture is None or snr_aperture <= 0:
        # Unknown SNR must NOT be read as worst-possible SNR. Defaulting to the ceiling meant a
        # frame whose SNR merely could not be *measured* was treated as maximally noisy, which
        # collapsed the Kalman gain and left the filter running open-loop while still accepting
        # every measurement. Fall back to the last confidently measured sigma instead; the
        # caller supplies it, and only when there is none do we use a mid-range default.
        random_sigma = (last_known_sigma_px if last_known_sigma_px is not None
                        else math.sqrt(floor_px * ceiling_px))
    else:
        random_sigma = calibration * fwhm_px / (2.0 * snr_aperture)
    random_sigma = min(ceiling_px, max(floor_px, random_sigma))

    bias = 0.0
    if clipped:
        bias = math.hypot(bias, clipped_bias_px)
    if saturated:
        bias = math.hypot(bias, saturated_bias_px)

    return min(ceiling_px, math.hypot(math.hypot(random_sigma, bias), unobservable_sigma_px))


@dataclass(frozen=True)
class KalmanParams:
    """Filter configuration.

    Attributes:
        process_noise_psd: Continuous white-noise acceleration PSD ``q``, in px^2/s^3. Set it to
            the **square of the acceleration the constant-velocity model must absorb**, not to a
            small number: ``q`` too low does not merely slow adaptation, it closes the validation
            gate on correct measurements. See ``config/default.json`` for the measurement. Sets
            how quickly the
            filter is willing to believe the target has changed velocity.
        initial_position_sigma_px: Initial position uncertainty.
        initial_velocity_sigma_px_s: Initial velocity uncertainty.
        gate_threshold: Mahalanobis-squared gate. 9.21 is chi-squared 99% for 2 DOF.
        law_calibration: Empirical correction to the accuracy law. See :data:`LAW_CALIBRATION`.
        sigma_floor_px: Lower clamp on derived measurement sigma.
        sigma_ceiling_px: Upper clamp on derived measurement sigma.
        unobservable_sigma_px: Sigma of disturbances the estimator cannot observe, chiefly
            camera jitter. See :func:`measurement_sigma`.
        max_sigma_to_p_ratio: Cap on measurement sigma as a multiple of the current position
            uncertainty. Keeps ``R`` commensurate with ``P`` so the validation gate retains
            discrimination. See :meth:`KalmanFilterCV.effective_sigma`.
        clipped_bias_px: Bias budget added when a detection is clipped.
        saturated_bias_px: Bias budget added when a detection is saturated.
    """

    process_noise_psd: float = 50.0
    initial_position_sigma_px: float = 50.0
    initial_velocity_sigma_px_s: float = 200.0
    gate_threshold: float = CHI2_99_2DOF
    law_calibration: float = LAW_CALIBRATION
    sigma_floor_px: float = 0.02
    sigma_ceiling_px: float = 20.0
    unobservable_sigma_px: float = 0.0
    max_sigma_to_p_ratio: float = 10.0
    clipped_bias_px: float = 2.0
    saturated_bias_px: float = 0.19


@dataclass(frozen=True)
class GateResult:
    """Outcome of a validation-gate test.

    Attributes:
        accepted: Whether the measurement passed the gate.
        mahalanobis_sq: Squared Mahalanobis distance of the innovation.
        nis: Normalised innovation squared -- the same quantity, named for its use as a filter
            consistency statistic. For a consistent 2-DOF filter its mean should be about 2.
        sigma_meas_px: Measurement sigma used for this update.
    """

    accepted: bool
    mahalanobis_sq: float
    nis: float
    sigma_meas_px: float


class KalmanFilterCV:
    """Constant-velocity Kalman filter over state ``[px, py, vx, vy]``.

    Attributes:
        params: Filter configuration.
    """

    def __init__(self, params: Optional[KalmanParams] = None) -> None:
        """Create an uninitialised filter.

        Args:
            params: Filter configuration.
        """
        self.params = params or KalmanParams()
        self._x = np.zeros(4, dtype=np.float64)
        self._P = np.eye(4, dtype=np.float64)
        self._initialised = False
        self._last_measured_sigma: Optional[float] = None

    # ---------------------------------------------------------------------------------------
    # State access
    # ---------------------------------------------------------------------------------------

    @property
    def initialised(self) -> bool:
        """Whether the filter holds a track."""
        return self._initialised

    @property
    def state(self) -> np.ndarray:
        """Current state vector ``[px, py, vx, vy]`` as a copy."""
        return self._x.copy()

    @property
    def covariance(self) -> np.ndarray:
        """Current state covariance as a copy."""
        return self._P.copy()

    @property
    def position(self) -> Tuple[float, float]:
        """Estimated position as ``(x, y)``."""
        return float(self._x[0]), float(self._x[1])

    @property
    def velocity(self) -> Tuple[float, float]:
        """Estimated velocity as ``(vx, vy)`` in pixels per second.

        This is the quantity the controller uses for feedforward, and the quantity its derivative
        term is taken from -- never a difference of raw measurements.
        """
        return float(self._x[2]), float(self._x[3])

    @property
    def position_sigma_px(self) -> float:
        """Scalar position uncertainty, the mean of the two positional standard deviations."""
        return float(0.5 * (math.sqrt(max(self._P[0, 0], 0.0))
                            + math.sqrt(max(self._P[1, 1], 0.0))))

    # ---------------------------------------------------------------------------------------
    # Filter operations
    # ---------------------------------------------------------------------------------------

    def initialise(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0) -> None:
        """Start a track at a measured position.

        Args:
            x: Initial x position.
            y: Initial y position.
            vx: Initial x velocity in pixels per second.
            vy: Initial y velocity in pixels per second.
        """
        self._x = np.array([x, y, vx, vy], dtype=np.float64)
        position_var = self.params.initial_position_sigma_px ** 2
        velocity_var = self.params.initial_velocity_sigma_px_s ** 2
        self._P = np.diag([position_var, position_var, velocity_var, velocity_var])
        self._initialised = True

    @staticmethod
    def transition(dt: float) -> np.ndarray:
        """Return the constant-velocity state-transition matrix for a step of ``dt``.

        Args:
            dt: Time step in seconds.

        Returns:
            The 4x4 matrix ``F``.
        """
        F = np.eye(4, dtype=np.float64)
        F[0, 2] = dt
        F[1, 3] = dt
        return F

    @staticmethod
    def process_noise(dt: float, psd: float) -> np.ndarray:
        """Return the continuous white-noise-acceleration process covariance.

        Args:
            dt: Time step in seconds.
            psd: Acceleration power spectral density ``q``.

        Returns:
            The 4x4 matrix ``Q``.
        """
        t2, t3 = dt * dt, dt * dt * dt
        Q = np.zeros((4, 4), dtype=np.float64)
        Q[0, 0] = Q[1, 1] = t3 / 3.0
        Q[2, 2] = Q[3, 3] = dt
        Q[0, 2] = Q[2, 0] = Q[1, 3] = Q[3, 1] = t2 / 2.0
        return psd * Q

    def predict(self, dt: float) -> Tuple[float, float]:
        """Advance the state by ``dt`` seconds.

        Args:
            dt: Time step in seconds. Must be non-negative.

        Returns:
            The predicted position as ``(x, y)``.

        Raises:
            ValueError: If ``dt`` is negative, or the filter is not initialised.
        """
        if dt < 0:
            raise ValueError(f"Time step must be non-negative, got {dt}")
        if not self._initialised:
            raise ValueError("Cannot predict before the filter is initialised")

        F = self.transition(dt)
        self._x = F @ self._x
        self._P = F @ self._P @ F.T + self.process_noise(dt, self.params.process_noise_psd)
        return self.position

    def innovation(self, z: Tuple[float, float],
                   sigma_meas_px: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the innovation and its covariance for a measurement.

        Args:
            z: Measured position as ``(x, y)``.
            sigma_meas_px: Measurement sigma.

        Returns:
            ``(innovation, innovation_covariance)``.
        """
        measurement = np.asarray(z, dtype=np.float64)
        predicted = self._x[:2]
        S = self._P[:2, :2] + np.eye(2) * (sigma_meas_px ** 2)
        return measurement - predicted, S

    def effective_sigma(self, sigma_meas_px: float) -> float:
        """Cap the measurement sigma relative to the current state uncertainty.

        A validation gate only discriminates while ``R`` is comparable to ``P``. Once ``R``
        dominates, the innovation covariance ``S = P + R`` is set almost entirely by ``R``, every
        normalised innovation becomes small, and the gate accepts anything -- it has stopped
        being a gate. Measured in that state: NIS sat at ~0 on frames whose estimate was drifting
        past 2000 px.

        Capping ``R`` at a multiple of ``P`` keeps the two commensurate, so the gate retains
        discrimination however the sigma was derived. This is deliberately a cap on the *ratio*
        rather than on coast duration: the failure that motivated it was not caused by coasting,
        it was a single-frame jump to the sigma ceiling.

        Args:
            sigma_meas_px: Proposed measurement sigma.

        Returns:
            The sigma actually used, capped relative to the current position uncertainty.
        """
        if not self._initialised or self.params.max_sigma_to_p_ratio <= 0:
            return sigma_meas_px
        position_sigma = self.position_sigma_px
        if position_sigma <= 0:
            return sigma_meas_px
        return min(sigma_meas_px, self.params.max_sigma_to_p_ratio * position_sigma)

    def gate(self, z: Tuple[float, float], sigma_meas_px: float) -> GateResult:
        """Test a measurement against the Mahalanobis validation gate.

        This is what rejects the impulse false positives the single-frame detector cannot
        eliminate: a salt cluster appears at an arbitrary position, so its innovation is large
        relative to ``S``, while a genuine detection sits within a few sigma of the prediction.

        Args:
            z: Measured position as ``(x, y)``.
            sigma_meas_px: Measurement sigma for this frame.

        Returns:
            A :class:`GateResult`.

        Raises:
            ValueError: If the filter is not initialised.
        """
        if not self._initialised:
            raise ValueError("Cannot gate before the filter is initialised")
        sigma_meas_px = self.effective_sigma(sigma_meas_px)
        nu, S = self.innovation(z, sigma_meas_px)
        distance = float(nu @ np.linalg.solve(S, nu))
        return GateResult(accepted=distance <= self.params.gate_threshold,
                          mahalanobis_sq=distance, nis=distance,
                          sigma_meas_px=sigma_meas_px)

    def update(self, z: Tuple[float, float], sigma_meas_px: float) -> None:
        """Fold a measurement into the state.

        Uses the Joseph form, which stays symmetric and positive-definite under finite precision
        where the shorter ``(I - KH)P`` form can drift negative over a long run.

        Args:
            z: Measured position as ``(x, y)``.
            sigma_meas_px: Measurement sigma.

        Raises:
            ValueError: If the filter is not initialised.
        """
        if not self._initialised:
            raise ValueError("Cannot update before the filter is initialised")

        H = np.zeros((2, 4), dtype=np.float64)
        H[0, 0] = H[1, 1] = 1.0
        R = np.eye(2, dtype=np.float64) * (sigma_meas_px ** 2)

        nu, S = self.innovation(z, sigma_meas_px)
        K = self._P @ H.T @ np.linalg.inv(S)
        self._x = self._x + K @ nu
        IKH = np.eye(4) - K @ H
        self._P = IKH @ self._P @ IKH.T + K @ R @ K.T

    def sigma_for(self, snr_aperture: Optional[float], fwhm_px: float,
                  clipped: bool = False, saturated: bool = False) -> float:
        """Derive the measurement sigma for a detection, using this filter's configuration.

        Args:
            snr_aperture: Measured aperture SNR.
            fwhm_px: Spot scale.
            clipped: Detection touches the frame edge.
            saturated: Detection core is saturated.

        Returns:
            Effective measurement sigma in pixels.
        """
        sigma = measurement_sigma(
            snr_aperture, fwhm_px,
            floor_px=self.params.sigma_floor_px, ceiling_px=self.params.sigma_ceiling_px,
            clipped=clipped, saturated=saturated,
            clipped_bias_px=self.params.clipped_bias_px,
            saturated_bias_px=self.params.saturated_bias_px,
            calibration=self.params.law_calibration,
            unobservable_sigma_px=self.params.unobservable_sigma_px,
            last_known_sigma_px=self._last_measured_sigma)
        if snr_aperture is not None and snr_aperture > 0:
            self._last_measured_sigma = sigma
        return sigma

    def reset(self) -> None:
        """Discard the track."""
        self._x = np.zeros(4, dtype=np.float64)
        self._P = np.eye(4, dtype=np.float64)
        self._initialised = False
