# Technical Design — Algorithms and Mathematical Models

This document is the algorithmic reference for implementation. Every formula here should have a
corresponding implementation with a docstring that cites the relevant section.

---

## 1. Domain background (needed for the Technical Evaluation Q&A)

In operational lasercom, PAT is a two-stage hierarchy:

- **Coarse stage** — wide-FOV camera (CMOS/CCD beacon camera) on a gimbal, brings the remote
  terminal into the acquisition region. Typical accuracy ~milliradians.
- **Fine stage** — fast steering mirror (FSM) driven by a quadrant photodiode / position
  sensitive detector, reaches microradian accuracy sufficient to couple the comms beam.

Illustrative published figures (useful for the report; cite them as indicative, not canonical):

- NASA LCRD optical module: ~4-inch telescope producing a ~15 µrad downlink beam, with a quadrant
  detector of roughly 2 µrad field of view.
- Patent-sourced coarse/fine hierarchies: gimbal coarse pointing ~3 mrad with FSM fine pointing
  ~5 µrad closed-loop; a smallsat FSO design quotes ~±1.6 mrad (3σ) coarse and ±80 µrad (3σ) fine.
- ESA SILEX terminals had to point to within ~2 µrad in the communication phase, with beams only
  a few µrad wide.
- **Real acquisition is slow**: SILEX reported maximum acquisition times up to ~130 s; NASA LCRD
  reports ~30–45 s; modelling work finds LEO links clustering near 50 s and interplanetary links
  near 200 s.

**Implication for the report:** the spec's ≤2 s acquisition target is an aggressive *simulation*
convenience, achievable because our virtual beacon is bright and the search region is a bounded
2000×2000 canvas rather than a µrad-scale sky search. Do not present ≤2 s as representative of
real on-orbit FSOC. Saying this explicitly demonstrates domain understanding.

**Coarse→fine handover criterion** in our simulation: the beacon is held within a defined lock
window around boresight for K consecutive frames with detection SNR above threshold. That is the
software analogue of "the target is now inside the fine sensor's FOV".

---

## 2. Beacon rendering

Model the beacon as a 2D Gaussian so that ground truth is known to sub-pixel precision:

```
I(x, y) = I0 * exp( -[ (x - xc)^2 + (y - yc)^2 ] / (2 * sigma^2) )
```

- FWHM = 2.355 · sigma. For a 10×10 px target use sigma ≈ 2–3 px.
- Render on a **supersampled grid (4×) and downsample** for anti-aliasing, so (xc, yc) can be
  non-integer and the ground truth centroid is exact.
  The downsample coordinate mapping is `x_out = (C + 0.5)/S - 0.5` (and likewise for rows),
  under the pixel-centre convention in `CLAUDE.md`. The natural-looking `x_out = C/S` is wrong
  and introduces a systematic `(S-1)/(2S)` px offset — 0.375 px at S=4. Phase 1 tests this
  mapping **in isolation to <0.01 px**, separately from the <0.05 px end-to-end render test,
  because a bias this size is large enough to invalidate the SNR curve but small enough to be
  absorbed by the centroid estimator and pass an end-to-end check.
- Shapes: `gaussian` (default for realism), `square` (spec default), `circle`.
  **Hard-edged shapes must be area-sampled, not point-sampled.** Testing `inside/outside` at each
  supersample centre quantises the shape's edge to the supersample grid, which caps centroid
  accuracy at `1/(2S)` px — 0.125 px at S=4, outside the 0.05 px budget. This is *not*
  recoverable by blurring afterwards: by the time a PSF is applied the sub-supersample edge
  position has already been discarded. Measured at S=4: point sampling gives 0.100 px worst-case
  centroid error and swings a 9×9 square's total intensity by 2.25 px² across sub-pixel phase;
  exact area coverage gives 0.000 px error and conserves intensity perfectly, even at S=1.
  Since the spec's *default* target shape is a square (parameter 9), this is on the main path,
  not an edge case. The square is separable so its coverage is exact; the circle uses a
  one-cell linear ramp at the boundary (measured ~0.003 px). An optional Gaussian PSF remains
  available for optical realism, but it is not what buys sub-pixel accuracy.
- Optional intensity decay: scale I0 over time to emulate changing range/attenuation.

**Sampling sanity check:** the Cramér–Rao bound for centroid estimation is minimised when the
pixel-to-spot-size ratio is roughly 1.5–2.5. A 10 px spot on unit pixels is comfortably
well-sampled, so we are not sampling-limited.

**Edge clipping is a real error source, and must not be silently absorbed.** A spot truncated by
the frame boundary has its centre of mass inside the surviving portion, so its measured centroid
is biased *inward*. Measured at the frame edge with the default beacon this reaches ~2 px —
40× the clean-render error — purely from truncation, with no noise involved. Consequences:

- Phase 1 accuracy is quoted only over frames where the spot is fully inside the viewport
  (margin ≥ 5σ); edge-clipped frames are reported separately, never pooled into the headline.
- Phase 3/4 must treat a detection near the frame boundary as lower-confidence: inflate the
  Kalman `R` for it (consistent with the SNR-derived `R` in section 6) rather than trusting it
  equally. A target drifting out of frame otherwise produces a smoothly growing, entirely
  systematic error that looks like tracker lag and is easy to misdiagnose.
- The detection result carries an explicit **`clipped`** flag, set when the blob's bounding box
  touches the frame edge. `R` inflation is the *response*; the flag is what makes the condition
  *diagnosable*. Telemetry logs it per frame so an error excursion can be attributed
  immediately rather than re-derived — the same reasoning as `from_fallback` on the resolved
  vision geometry.

**Saturation is the sibling failure, and gets the same treatment.** Once atmospheric brightening
or scintillation drives peak intensity to the 8-bit ceiling, the spot's top is flattened and the
intensity-weighted centroid biases toward the *geometric* centre of the saturated region rather
than the photometric centre of the spot. The failure shape is identical to clipping — smooth,
systematic, noise-free, and easily mistaken for control lag — so it gets a **`saturated`** flag
alongside `clipped`, logged per frame, with the same `R` inflation in response. Phase 2 pins a
test quantifying centroid bias against saturation fraction, which is what sets the flag's
threshold rather than a guess.

Note the asymmetry with noise: clipping and saturation are *bias* sources, whereas shot and read
noise are *variance* sources. Averaging more frames reduces the latter and does nothing for the
former, which is precisely why they need flagging rather than just a larger `R`.

**Phase 2 measured the saturation bias curve, and the result is more nuanced than expected.**
Clipping a *symmetric* spot *symmetrically* preserves its centroid exactly, so saturation alone
does not produce a large bias — the unsaturated wings retain the sub-pixel information and
dominate the intensity-weighted sum. Bias appears only as the estimator discards those wings.
Worst-case bias across sub-pixel phase, default 10 px Gaussian beacon
(`tests/test_saturation.py`):

| Estimator regime | What survives thresholding | Worst bias |
|---|---|---|
| Unthresholded centre of gravity | everything | < 0.005 px |
| Threshold at 0.7 × peak | bright core plus partial edge pixels | < 0.06 px |
| Threshold at the ceiling (degenerate) | fully saturated pixels only | ~0.19 px |

The degenerate case is the mechanism originally anticipated: with only ceiling-valued pixels
left, the centroid is the geometric centre of an *integer pixel set* and quantises to the pixel
grid.

**Conclusion: saturation costs precision, not lock.** Even fully saturated, ~0.19 px sits about
50× inside the 10 px tracking requirement. It matters for sub-pixel accuracy claims and for the
SNR curve; it does not threaten lock retention. So the `saturated` flag should **inflate `R`
modestly rather than reject the detection** — discarding a 0.19 px-biased measurement costs far
more than keeping it. This is the opposite of the right response to a salt-and-pepper false
positive, which must be rejected outright, and the distinction is worth making explicitly in the
Q&A: a *bounded bias* is still information, an *outlier* is not.

---

## 3. Trajectory models

All trajectories are parametric functions of time returning `(x, y)` in canvas coordinates.

| Motion | Equations |
|---|---|
| Straight line | `x = x0 + vx*t`, `y = y0 + vy*t` |
| Circular | `x = xc + R*cos(w*t)`, `y = yc + R*sin(w*t)` |
| Figure-of-8 (Lissajous) | `x = A*sin(a*w*t + d)`, `y = B*sin(b*w*t)` with `a:b = 1:2` |
| Sinusoidal | `x = x0 + vx*t`, `y = y0 + A*sin(w*t)` |
| Archimedean spiral | `r = b*theta`, `x = r*cos(theta)`, `y = r*sin(theta)` |
| Random walk | `x[k+1] = x[k] + xi`, `xi ~ N(0, sigma_step^2)` |
| Ornstein–Uhlenbeck (preferred stochastic model) | `x[k+1] = x[k] + theta*(mu - x[k])*dt + sigma*sqrt(dt)*eps` |

The Archimedean spiral doubles as the **acquisition search pattern**. Set arm spacing
`D = 2*pi*b` equal to the beacon width so the scan fully covers the uncertainty region — this
mirrors the operational practice of setting spiral line spacing to the beam divergence diameter.

Mandatory motions: straight line, circular, figure-of-8, random. The rest are optional bonus.

**Boundary handling:** clamp, bounce, or wrap at canvas edges — make it a config option, default
bounce. Document what happens when the target reaches an edge.

---

## 4. Noise and disturbance models

All implemented as pure functions `(frame, params) -> frame`, composable in a configurable order.

### 4.1 Sensor noise

- **Additive white Gaussian noise (AWGN):** `frame + rng.normal(0, sigma, shape)`, clipped to
  [0, 255]. Spec allows sigma up to 20.
- **Poisson (shot) noise:** `rng.poisson(frame.astype(float))`. Signal-dependent and physically
  correct for photon counting — brighter regions get more absolute noise.
- **Salt & pepper:** select a random mask covering up to 10% of pixels; set half to 0 and half to
  255. This is the **primary false-positive threat** to centroiding and the main reason a median
  prefilter is mandatory.
- Optional extras for realism: read noise (fixed-sigma additive Gaussian), dark current (Poisson
  offset), fixed-pattern noise (fixed multiplicative map).

### 4.2 Atmospheric degradation — Koschmieder model

The standard atmospheric scattering model used in computer vision:

```
I(x) = J(x) * t(x) + A * (1 - t(x))
```

where `J` is the clear image, `A` is atmospheric light (airlight), and `t` is transmission. Under
Beer–Lambert attenuation `t(x) = exp(-beta * d(x))`.

For our 2D scene use a spatially uniform (or gently graded) `t`, with `beta` as the user's
"reduction in contrast and brightness" knob:

| Preset | Approx. behaviour |
|---|---|
| Clear | `beta ≈ 0`, `t ≈ 1` |
| Haze | mild `beta`, high `A` |
| Fog | strong `beta`, high `A`, plus mild Gaussian blur |
| Rain | moderate `beta` plus sparse streak overlay |
| Low light | scale `J` down, then apply Poisson + read noise |

This is a per-pixel affine blend — trivially fast, comfortably real-time at >30 FPS.

### 4.3 Turbulence / scintillation

Modulate `I0` over time by a random factor:
- **Weak turbulence:** log-normal distribution.
- **Moderate–strong:** Gamma–Gamma distribution, with α, β derived from the Rytov variance
  `sigma_R^2` via the standard **plane-wave, point-receiver** relations:

  ```
  alpha = 1 / (exp(0.49 * s2 / (1 + 1.11 * s2^(6/5))^(7/6)) - 1)
  beta  = 1 / (exp(0.51 * s2 / (1 + 0.69 * s2^(6/5))^(5/6)) - 1)
  ```

  Values these actually produce (`src/noise/turbulence.py`, verified against the closed-form
  scintillation index in tests):

  | `sigma_R^2` | α | β | scintillation index |
  |---|---|---|---|
  | 0.1 | 21.59 | 19.82 | 0.099 |
  | 1.0 | 4.39 | 2.56 | 0.706 |
  | 4.0 | 4.34 | 1.31 | 1.170 |
  | 10.0 | 5.69 | 1.10 | 1.243 |

  **Correction:** an earlier draft of this section quoted α≈2.95/β≈2.46 at `sigma_R^2 = 1` and
  α≈2.48/β≈0.98 at 10. Those do not follow from the relations above at any parameterisation and
  have been replaced by the computed values. Published α, β tables vary widely because they
  depend on the wave model (plane vs. spherical), on aperture-averaging, and on whether the
  table is keyed by `sigma_R` or `sigma_R^2` — so **quote the formula, not a remembered table**,
  and if challenged in Q&A cite the scintillation index, which is model-independent and is what
  our tests actually assert:

  ```
  SI = var(I)/mean(I)^2 = 1/alpha + 1/beta + 1/(alpha*beta)
  ```

  Note α is non-monotonic in `sigma_R^2` while SI saturates near ~1.2. That is the physically
  expected saturation regime, not a numerical artefact.
- **Beam wander:** a slow random offset added to the true target position (an Ornstein–Uhlenbeck
  process works well).

Keep this as a cheap multiplicative time series. **Do not** implement full GPU phase-screen
propagation — the compute cost is not justified for coarse alignment simulation.

### 4.4 Mechanical disturbances

- **Camera jitter:** ±20 px/frame random offset applied to the viewport extraction position.
- **Platform motion:** a slow drift of the camera boresight following a selectable trajectory
  (linear mandatory; circular/random/spiral/figure-8 optional), bounded at ±20 px/frame.

Note the distinction: jitter is high-frequency zero-mean; platform motion is a low-frequency
bias the controller must actively reject.

---

## 5. Detection and centroiding

### 5.1 Pipeline order

```
frame
  -> median filter (3x3)            # kills salt & pepper before anything else
  -> top-hat morphology             # suppresses smooth background, enhances small bright spots
  -> adaptive threshold             # NO fixed constants; default mean + k*sigma on the residual
  -> connected components / blob    # candidate list
  -> size + shape gating            # reject noise specks and oversized blobs
  -> thresholded intensity-weighted centroid on best candidate
  -> Kalman validation gate         # reject association outliers
```

### 5.1.1 Choice of threshold operator (default: `mean + k*sigma`)

The default is `mean + k*sigma` (k ≈ 3) computed **on the top-hat residual**, not Otsu.

The defensible argument is about **fill factor in the raw histogram**: a 10×10 beacon occupies
~0.03% of a 640×480 frame, so the raw-frame histogram is overwhelmingly unimodal background.
Otsu maximises inter-class variance under an implicitly bimodal model, and at that fill factor
there is no second mode for it to find — the split lands somewhere inside the noise distribution
and is governed by the background's own shape rather than by the target.

**Correction (Phase 3, measured).** An earlier draft of this section said Otsu applied to the
*top-hat residual* rather than the raw frame is "much better behaved". Measurement does not
support that. On Gaussian noise alone, with no impulses:

| noise sigma | Otsu candidates | Otsu centroid error | default operator error |
|---|---|---|---|
| 0 | 1 | 0.03 px | 0.18 px |
| 2 | 1 | 0.03 px | 0.05 px |
| 5 | 1755 | 291 px | 0.16 px |
| 10 | 2501 | no valid detection | 0.37 px |

Otsu on the residual is excellent when noise is negligible and fails completely once noise is
realistic. The fill-factor argument bites on the residual too: a 10×10 beacon is ~0.03% of the
frame, so once noise is present the residual histogram is dominated by noise and Otsu splits the
*noise* distribution rather than separating spot from background. It remains selectable, and the
table above is precisely the comparison row we want in the report — but it is not a viable
default, and the claim in Q&A should be the measured one above rather than the softer hedge.

Selectable operators, all parameter-free or single-parameter: `mean_plus_k_sigma` (default),
`otsu`, `adaptive_gaussian`, `percentile`. Benchmark all four on the SNR sweep and put the table
in the report.

### 5.1.2 Spot-scale bootstrap (the Benchmark-2 keystone)

`resolve_geometry(fwhm_px)` needs the spot scale; the scale can only be measured from a
detection; detection geometry depends on the scale. Resolved as a three-tier design.

**Tier 0 — bootstrap, strictly scale-free.** 3x3 median -> per-frame normalise -> high-percentile
threshold -> connected components -> area gate -> second-moment FWHM of the strongest survivor.
Every constant is scale-free: the median kernel is 3 because impulses are one pixel wide (not
because the spot is any size), the percentile is dimensionless, and the gates are a sampling
floor plus a fraction of frame dimension.

The **area gate is load-bearing, not hygiene.** A 3x3 median fails wherever >=5 of 9 pixels are
corrupted; at 10% independent salt-and-pepper that is `P ~ 8.9e-4`, leaving ~273 residual impulse
pixels on a 640x480 frame — against a 99.9th-percentile selection of ~307 pixels. Without the
area gate, bootstrap locks onto salt. Residual impulses are 1–2 px; the smallest legal beacon is
5 px across (~20 px^2 area), so the gate has margin.

> **The 273-pixel figure assumes *independent* impulses, and H.264 violates that.** Compression
> smears impulses across DCT blocks, producing *correlated multi-pixel* artifacts. The direction
> of the risk is unfavourable: compression makes residual artifacts **larger**, pushing them out
> of the 1–2 px class the area gate cleanly rejects and toward the beacon's own size class.
> The gate is therefore derived here from the independent case and **must be re-validated
> against H.264 round-tripped video in Phase 6** — if artifacts reach the beacon's size class,
> area alone stops separating them and the gate needs a shape or contrast discriminant added.

**Tier 1 — confirmation via scale-normalised Laplacian of Gaussian, on the Tier-0 ROI only.**

**Do not use top-hat for scale selection.** An earlier draft of this design did, and measurement
shows it cannot work: top-hat response is **monotonically non-decreasing** in kernel size once
the structuring element exceeds the spot, because a larger element simply passes the whole spot
plus more background. There is no interior maximum to find. Measured on a sigma=2.5 spot over a
x1.3 ladder, the argmax landed at k=19 with a peak-to-runner-up ratio of **1.000** — a perfectly
flat top. Top-hat remains correct for *background suppression* in the detection path; it is the
wrong operator for *scale selection*.

The correct operator is Lindeberg's scale-normalised LoG, `sigma^2 * |laplacian(G)|`, which
attains a genuine maximum over scale at the blob's characteristic scale. Measured recovery on a
x1.3 sigma ladder with sigma_noise = 10:

| true spot sigma | recovered | after parabola refinement |
|---|---|---|
| 1.5 | 1.69 | 1.65 |
| 2.5 | 2.86 | 2.57 |
| 5.0 | 4.83 | 4.91 |
| 2.5 (SNR ~ 3) | 2.20 | 2.50 |

**Confidence is curvature, not peak-to-runner-up.** The ratio check does not merely fail to
discriminate a broad plateau — it is *inverted*. Measured: 1.01–1.06 for correct estimates on
real spots, 1.12–1.14 for pure noise with no spot at all. A x1.3 ladder is fine enough that
adjacent rungs are always close in response, so the ratio carries no information about how well
determined the scale is. Instead, fit a parabola to `log(response)` vs `log(sigma)` at the peak:

- **Curvature** (`-2a` from the fit) is the confidence measure. Measured ~1.9–2.4 for real spots
  across the SNR range, **0.0** for pure noise. Clean separation.
- The same fit gives **sub-rung refinement** for free, and it is what pulls 2.20 -> 2.50 in the
  low-SNR row above.

**Shape dependence: measured, and absorbable.** Both Tier-0 and Tier-1 conversions are
Gaussian-derived — half-maximum area to FWHM assumes a circular half-max disc, and the LoG
extremum location is derived for a Gaussian profile. The spec's *default* target shape is a
square (parameter 9), and in Mode B the shape is unknown, so any bias must be absorbed rather
than looked up. Measured, reporting Tier-0 FWHM divided by nominal target size:

| nominal L | gaussian | square | circle | spread |
|---|---|---|---|---|
| 6 | 1.030 | 1.064 | 0.959 | 10.9% |
| 10 | 1.022 | 1.106 | 1.003 | 10.2% |
| 14 | 1.016 | 1.117 | 0.994 | 12.4% |
| 20 | 0.998 | 1.123 | 0.992 | 13.2% |

**Verdict: (a), absorbable inside the 30% agreement band.** The bias is systematic and
explainable rather than noise: a square's half-maximum region *is* the square, of area `L^2`, so
the disc-equivalent conversion returns `2*sqrt(L^2/pi) = 1.128*L`. The measured square/Gaussian
ratio converges to 1.125 at L=20 — analytic and measurement agree to 0.3%. A circle *is* a disc,
so it is unbiased by construction. It does, however, consume ~13% of a 30% budget, leaving less
headroom for noise-driven error than the band suggests; that is worth stating rather than
treating 30% as all available.

**A shape-agnostic joint estimator was tested and rejected.** Matching an equivalent-Gaussian
sigma on integrated flux *and* half-max area jointly does not clearly beat area alone (18.0% vs
23.5% spread in an isolated comparison), and it reintroduces the exact failure mode that killed
the second-moment estimator: flux integrates over large radius, where noise dominates and signal
does not. Half-max area is preferred precisely because it only counts pixels near the peak.

**Contrast invariance (the Mode B property that matters most).** Reported scale is *completely*
insensitive to atmospheric extinction — spread is identical to three decimal places from clear
(`beta=0`) through heavy fog (`beta=0.75`), because both the percentile threshold and the
half-maximum contour are defined relative to the spot's own peak rather than to any absolute
level. An evaluator video of unknown haze or fog cannot shift our geometry.

**The blur hypothesis is refuted, and interestingly so.** Softening a square *does* pull its
ratio toward the Gaussian (1.095 → 0.967 as blur sigma goes 0 → 2.5), so the shapes converge as
predicted. But the overall *spread* gets worse, not better (11.7% → 17.8%), because blur inflates
the **Gaussian's own** measured scale (1.019 → 1.099). That is correct behaviour, not estimator
failure: a blurred Gaussian genuinely is wider, `sigma_total = sqrt(sigma_spot^2 + sigma_blur^2)`,
and the measured growth matches quadrature to within 1.5%. So the report line is the opposite of
the intuition: **the shape bias is *not* worst in clear conditions** — clear is the best case at
11.7%, and blur makes shape disagreement worse rather than washing it out. Within our configured
presets the worst is fog at blur 1.5 (13.6%); only at a blur of 4.0, far beyond any preset, does
the spread reach 27.2% and approach the band.

An earlier reading of the preset sweep attributed fog's larger spread to reduced SNR. Isolating
the variables shows that is wrong: contrast alone moves the spread not at all, and the entire
effect is the blur term.

#### Tier-1 shape bias, and how the agreement band is defined

The LoG extremum at `t = sigma^2` is Gaussian-derived just as the half-max-area conversion is, so
Tier 1 carries its own shape offset. Measured on identical frames, reported FWHM / nominal L:

| L | shape | Tier 0 | Tier 1 | Tier1/Tier0 |
|---|---|---|---|---|
| 10 | gaussian | 1.009 | 1.018 | 1.009 |
| 10 | square | 1.106 | 0.959 | 0.868 |
| 10 | circle | 0.997 | 0.848 | 0.851 |
| 14 | gaussian | 1.019 | 1.017 | 0.998 |
| 14 | square | 1.117 | 0.950 | 0.851 |
| 14 | circle | 1.000 | 0.853 | 0.853 |
| 20 | gaussian | 1.005 | 1.006 | 1.002 |
| 20 | square | 1.123 | 0.958 | 0.853 |
| 20 | circle | 1.000 | 0.847 | 0.847 |

The two estimators have **different and opposed** shape biases. Tier 0 reads a square *high*
(+11.7%) and a circle correctly; Tier 1 reads a square slightly low (-4.5%) and a circle
substantially low (-15%). Tier 1's circle offset is analytic: the normalised LoG of a disc of
radius `R` peaks at `sigma = R/sqrt(2)`, giving `FWHM_eq = 0.833*L` against a measured 0.847.

**Removing "each estimator's Gaussian-relative offset" turns out to be a no-op.** Both estimators
already return a Gaussian's true FWHM — `Tier1/Tier0 = 1.001` on Gaussians. There is no
Gaussian-relative offset left to subtract. The ~15% divergence appears *only* on non-Gaussian
profiles, so it cannot be calibrated away with a shape-independent constant.

**But the divergence is not noise — it is information.** The ratio `rho = Tier1/Tier0` is stable
and bimodal. Measured over 216 samples spanning 4 sizes x 3 blur levels x 3 noise levels x 2
seeds:

| cluster | mean | std | range |
|---|---|---|---|
| Gaussian-like | 1.001 | 0.015 | [0.959, 1.034] |
| hard-edged (square and circle) | 0.865 | 0.021 | [0.835, 0.934] |

The clusters are **separable, with a gap of +0.024**. Square and circle are indistinguishable
from each other (0.866 vs 0.865), so `rho` is a *profile-class* discriminant — smooth versus
hard-edged — not a shape identifier. That is all we need, and it works without knowing the shape,
which is the Mode B constraint.

**Decision: option (ii), implemented as cluster membership rather than offset subtraction.**

1. Tier 0's value is the reported scale. It is the more noise-robust of the two (half-max area
   only counts pixels near the peak), and it is already validated at 0.4-10.7%.
2. Tier 1 is the cross-check. Compute `rho = Tier1_FWHM / Tier0_FWHM`.
3. If `rho` lands inside a known cluster -- `[0.94, 1.10]` or `[0.80, 0.94)` -- the disagreement
   is **explained by profile class** and does *not* set `from_fallback`.
4. If `rho` lands outside both clusters, the disagreement is **unexplained** and `from_fallback`
   is set.

This keeps the band meaning exactly what it should: *these two disagree for a reason we do not
understand*. Widening the band to ~35% to swallow the shape divergence would have made
`from_fallback` insensitive at precisely the moment we most want it firing.

**Three honest caveats.**

- **The gap is narrow.** +0.024 separated cleanly across 216 samples, but that is a 2.4% margin,
  not a comfortable one. The classification must therefore **fail safe**: `rho` landing *in the
  gap* counts as unexplained, not as a near-miss to be rounded to the closer cluster.
- **Only three profiles have been tested.** An evaluator's beacon could be a defocused annulus or
  a saturated flat-top, which may land anywhere on the `rho` axis. The fail-safe rule above is
  what makes that survivable: an unrecognised profile yields `from_fallback` and configured
  geometry, rather than a confident wrong scale.
- **The clusters must be re-validated against H.264 round-tripped video in Phase 6**, the same
  caveat that applies to the area gate. Compression alters edge sharpness, which is precisely the
  property `rho` keys on, so the cluster centres may move.

#### Correction: `rho` is advisory, not a classifier — the band stays wide

The Tier-1/Tier-0 ratio initially looked like a profile-class discriminant. A 216-sample sweep
(4 sizes, all >= 8 px, one brightness) put the smooth cluster at 1.001 +/- 0.015 and the
hard-edged cluster at 0.865 +/- 0.021, separating with a gap of **+0.024**.

**A wider sweep refuted that.** Over 10 sizes from 5 to 24 px, 4 noise levels and 4 seeds:

| profile | mean | std | range |
|---|---|---|---|
| gaussian | 0.987 | 0.033 | [0.868, 1.054] |
| square | 0.876 | 0.029 | [0.843, 0.976] |
| circle | 0.861 | 0.040 | [0.822, 1.072] |

Raw gap **−0.204**: 99 hard-edged samples sit above the Gaussian minimum, 144 Gaussian samples
below the hard-edged maximum. Smoothing does not rescue it — run means over 1, 5, 10 and 20
frames all leave a gap near −0.05, because the spread is **systematic in spot size**, not
frame-to-frame noise, so averaging cannot remove it. The original sweep simply did not span small
spots.

**So the decision reverts to option (i): a wide band, with the budget stated explicitly.**

- shape divergence between the two estimators: up to ~15% (0.987 vs 0.861);
- spread from spot size and noise: ~±11% at 3 sigma;
- total ~26%, so the band is set at **±30%**, `rho in [0.70, 1.30]`.

The concern that widening blunts `from_fallback` at the worst moment is real and stands. The
mitigation is that **`rho` is the weakest of five gates and is weighted accordingly**. The sharp
ones all discriminate cleanly and carry the load:

| gate | discrimination |
|---|---|
| saturation gate | rejects at >10% core saturation; catches the +173% inflation case the clamp misses |
| Tier-0 peak significance | real beacons 6.1–40.0 sigma, pure noise 3.4–3.9 sigma |
| Tier-1 curvature floor | real spots ~1.9–2.4, pure noise 0.0 |
| Tier-1 ladder-edge check | argmax at a ladder end carries no information |
| `rho` band | ±30%, absorbs shape; gross disagreement only |

`rho` is still logged **per frame**, and the advisory profile label with it. It remains a genuine
diagnostic trace even though it is not a decision input: a `rho` trajectory that drifts across
runs is evidence about what changed, and a bare `from_fallback` with no history is not.

#### Saturation migrates a smooth beacon toward hard-edged

Saturated flat-top is not hypothetical — scintillation plus a bright atmospheric preset produces
it in normal operation. Measured on a Gaussian beacon as peak intensity rises:

| peak | saturated fraction | Tier-0 scale error | `rho` |
|---|---|---|---|
| 200 | 0.000 | +2% | 1.009 |
| 255 | 0.068 | +5% | 0.971 |
| 300 | 0.152 | +16% | 0.935 |
| 600 | 0.291 | +53% | 0.906 |

`rho` walks steadily downward as the top flattens — a genuinely smooth beacon drifting toward the
hard-edged region mid-run, which is one more reason the label cannot gate anything. More
importantly the **scale error** reaches +173% at 45% saturation, and the `max_fwhm` clamp is not
a safety net: a 6 px beacon at that level reports 16.4 px, stays under the clamp, and claims
success. Only the explicit saturation gate catches it.

**Tier 2 — continuous refinement.** While locked, the ROI second moment is essentially free (the
moments are already being computed for the centroid) and feeds the configured EMA.

#### Tier-2 costs, and the ratchet

**Re-check cost is set by measurement, not by a round number.** Full-frame Tier 0 spans 17 ms at
640×480 to 234 ms at 2000×2000 — a 14× range, so no single interval could be right for both. But
once locked we know roughly where the target is, so a periodic re-check has no reason to sweep
the canvas. Running Tier 0 on a padded ROI plus the already-cropped Tier 1 gives:

| frame | full-frame Tier 0 | steady-state ROI re-check | speed-up |
|---|---|---|---|
| 640×480 | 17.0 ms | 2.53 ms | 7× |
| 1920×1080 | 110 ms | 2.12 ms | 52× |
| 2000×2000 | 234 ms | 2.45 ms | 96× |

Steady-state cost is now **~2.5 ms at every frame size**, so the interval barely needs to vary
once locked; `recommended_recheck_interval_s` derives it from measured cost against a duty
budget (default 2%), clamped to [0.5 s, 5 s]. Full-frame is reserved for **cold bootstrap and
post-loss re-acquisition only**, which is the difference between 11% and 0.1% of the 2 s
acquisition budget.

A real bug surfaced here: the frame-fraction gates were being computed from the *crop*, so on a
128 px ROI the `max_fwhm` ceiling collapsed to 6.4 px and rejected a legitimate 14 px beacon.
Gates describe "small relative to the scene", and the scene is the full frame however much of it
we chose to examine — `bootstrap_scale` now takes an explicit `gate_shape`.

**The ratchet.** Growing the assumed scale when detection fails is necessary: too small a scale
puts the top-hat element inside the spot and deletes the beacon, and that failure is
self-reinforcing. But growth plus the integer deadband can interact into a one-way ratchet —
each failure nudges the scale up, the deadband lets accumulated growth take effect, and nothing
symmetric pushes it back down.

The defence is to keep two scales apart:

- **`estimated_fwhm_px` is persistent** and is written *only* by valid measurements. Failures
  never touch it.
- **`search_fwhm_px` is transient**. It inflates while blind (capped at 3×) and resets to the
  estimate the instant any candidate reappears.

Because growth lives only in the transient quantity, a 200-cycle run with intermittent failures
at constant true scale returns to the correct resolved kernel instead of drifting upward.

**Saturation gating is a second path into the same ratchet, and needs a different answer.** A
gated stretch looks like sustained failure to a naive grow policy — but it is not. Saturation,
an out-of-band `rho`, or low curvature all leave us with a perfectly good detection whose scale
we decline to trust: we can see the target, we simply cannot measure it. So *blocked* frames
**hold** the scale and never inflate the search, while *blind* frames (no candidate at all)
inflate it. The two are counted separately in the trace, because they mean different things and
a reader needs to tell them apart.

**Re-check triggers.** Periodic (~2 s); smoothed FWHM drifting >25% from the value geometry was
last resolved at; detection quality degrading; and unconditionally after any loss. Re-resolving
needs a deadband — resolved geometry yields *integer* kernel sizes, so an FWHM oscillating near a
boundary would chatter the kernel frame to frame and shift the detection statistics underneath
the tracker. Re-resolve only when a resolved integer would move by >=2.

**Anti-poisoning.** Never let one frame set the scale: require M-of-N agreement (3 of 5 frames
within 25%) before locking in.

> **The parallel fallback A/B is conditional, not unconditional.** Running fallback geometry
> alongside estimated geometry to compare detection quality costs an extra threshold pass at
> precisely the moment we are fighting a 2 s acquisition budget against an 11.6 s worst-case
> search (section 7.5). So: **measure the cost first**, and gate the A/B on Tier 0 and Tier 1
> *disagreeing* (outside the 30% agreement band) rather than running it on every acquisition.
> When the two estimators agree, there is nothing for the A/B to arbitrate.

**`from_fallback` fires when:** no candidate survives Tier 0; Tier 0 and Tier 1 disagree in a way
*not explained by profile class* -- see the `rho` cluster rule above, which replaces the earlier
flat 30% band;
the estimate hits a clamp bound; the LoG argmax sits at the **edge of the ladder** (true scale is
outside the search range, so the argmax is meaningless — this is what catches the sigma=8 and
sigma=15 cases above); or curvature falls below the confidence floor.

**The asymmetry that shapes the design.** Underestimating scale is catastrophic; overestimating
is merely lossy. Too small a scale puts the top-hat structuring element *inside* the spot, so the
opening deletes the beacon — and the failure is self-reinforcing, because no detection means no
scale correction. Round resolved kernels **up**, and on detection failure **grow** the assumed
scale rather than shrink it.

**Validation.** Mode A's known scale is the test oracle — this is why the scale is *estimated* in
both modes rather than declared in Mode A. Declaring it would leave the estimator exercised only
in the mode where we cannot check it, which is precisely the 30% benchmark. Sweep estimator error
against ground truth over SNR, **frame resolution** and **spot size**, plus two adversarial runs:

- **Approaching target**, true scale doubles mid-run.
- **Receding target**, true scale halves mid-run. **This is the case that will find the bug.**
  Growing scale is the safe direction and rides the round-up bias and the grow-on-failure policy
  downhill; it will probably pass on the first attempt. Shrinking scale runs *against* both
  policies, which actively resist tracking the true value downward. Assert that the estimate
  converges to the new smaller scale within a bounded number of frames rather than sticking high.

---

### 5.2 Centroid estimators

**Centre of Gravity (CoG):**
```
xc = sum(x * I) / sum(I)
yc = sum(y * I) / sum(I)
```

**Thresholded CoG:** subtract a background threshold T before summing. Use `T = mean + k*sigma`
with k ≈ 3, or Otsu. This is essential — unthresholded CoG is badly biased by background and
impulse noise.

**Iteratively-weighted CoG (IWCoG):** recentre the window on the current estimate and re-weight,
iterating 2–3 times. Produces lower variance than plain CoG at low SNR.

**2D Gaussian fit / parabolic peak interpolation:** fit a Gaussian (or parabola to the log of the
peak neighbourhood) for sub-pixel refinement. Remains unbiased to lower SNR than simple centroid
methods, at higher compute cost. Use as an optional refinement stage.

### 5.3 Accuracy laws (quote these in the report and Q&A)

- Shot-noise-limited: `sigma_centroid ≈ sigma_PSF / sqrt(N_photons)` — precision scales as the
  inverse square root of photon count in the shot-noise-limited case, and as the inverse of
  photon count in the background-noise-limited case.
- General astrometric rule of thumb: `sigma_x ≈ FWHM / (2 * SNR)`, with a Gaussian-PSF constant
  around 0.6–0.67.
- Near-CRLB estimators reach roughly 1/100 pixel at ~1000 detected photons; practical star
  trackers assume ~0.1 px.
- Systematic "peak-locking" bias tends to dominate random error above SNR ≈ 10.

**Consequence:** the ≤10 px requirement has 2–3 orders of magnitude of margin when locked. The
real challenge is *robustness at low SNR*, not raw precision. The interesting engineering
question — and the one to present — is where the pipeline breaks down (SNR ≈ 1–2, where centroid
error approaches the spot half-width), not how precise it is when conditions are good.

### 5.4 Why not OpenCV's built-in trackers as primary

KCF / CSRT / MOSSE are appearance-based trackers designed for textured objects. For a symmetric
bright dot, detection-by-thresholding plus centroiding is both faster and more accurate, and the
built-ins reacquire poorly after occlusion. In published benchmarks KCF runs at hundreds of FPS
while CSRT is more accurate but far slower. Keep them as an optional redundancy/comparison path
for the report, not as the core.

---

## 6. State estimation — Kalman filter

Constant-velocity model. State `x = [px, py, vx, vy]^T`.

**Prediction:**
```
x_pred = F @ x
P_pred = F @ P @ F.T + Q

F = [[1, 0, dt, 0],
     [0, 1, 0, dt],
     [0, 0, 1,  0],
     [0, 0, 0,  1]]
```

**Continuous white-noise process covariance** with PSD `q`:
```
Q = q * [[dt^3/3, 0,      dt^2/2, 0     ],
         [0,      dt^3/3, 0,      dt^2/2],
         [dt^2/2, 0,      dt,     0     ],
         [0,      dt^2/2, 0,      dt    ]]
```

**Measurement** `z = [px, py]` (the centroid), `H = [[1,0,0,0],[0,1,0,0]]`, `R = sigma_meas^2 * I2`.

**`R` must be adaptive, derived per frame from the measured detection SNR** (Phase 4 item; flag
it in the report as an innovation point). Section 5.3 already gives the law:

```
sigma_meas ≈ FWHM / (2 * SNR)        # floored at a sub-pixel minimum, capped at the ROI radius
```

A *static* `R` mis-scales the Mahalanobis validation gate exactly when it matters most. At low
SNR the true measurement scatter grows but a fixed small `R` keeps the gate tight, so genuine
detections are rejected and we drop lock — directly attacking the <5% loss-rate requirement. At
high SNR the same fixed `R` is too loose, and salt-and-pepper false positives pass the gate.
Deriving `R` from a quantity we already compute costs essentially nothing and makes both the gate
and the innovation covariance self-consistent across the whole SNR range. Validate on the SNR
sweep: the normalised innovation squared (NIS) should sit inside its chi-squared bounds at every
SNR, which is the standard consistency check and a strong Q&A answer.

**Update:**
```
K = P_pred @ H.T @ inv(H @ P_pred @ H.T + R)
x = x_pred + K @ (z - H @ x_pred)
P = (I - K @ H) @ P_pred
```

**Three uses, all essential:**
1. **Predictive feedforward** to the controller — compensates the structural one-frame
   measurement lag inherent to any feedback tracker following a moving reference.
2. **Validation gating** — Mahalanobis distance rejects salt-and-pepper false positives.
3. **Coasting** through dropouts/occlusion, enabling the ≤1 s re-acquisition requirement.

Use a constant-acceleration model or an IMM (CV + CA + coordinated turn) for the circular and
figure-8 maneuvering cases if time permits.

**Multi-target (optional):** nearest-neighbour or global nearest-neighbour data association, with
M-of-N track initiation/confirmation/deletion logic.

---

### 6.1 Adaptive R, calibrated against measurement

`sigma_meas = FWHM / (2 * SNR_aperture)` is the *theoretical* law. Measured against the Phase 3
SNR sweep it understates our actual per-axis error: comparing law to measurement across three
shapes and four SNR bands gives a median ratio of **1.32** (spread 0.74–2.25 once the saturated
high-SNR bins, where the law breaks down entirely, are excluded). The implementation carries that
factor as `LAW_CALIBRATION`. Using the raw law would make `R` optimistic by a third, tightening
the validation gate and rejecting genuine detections — precisely the failure adaptive `R` exists
to prevent. The spread is wide enough that this is a calibration, not a constant of nature.

**Bias sources are folded in as variance, deliberately.** `clipped` and `saturated` mark a known,
bounded, systematic displacement rather than extra noise. A Kalman filter has no representation
for bias, so `R = sigma_meas^2 + bias_budget^2`, with the budgets taken straight from the
measurements that produced them (2 px clipping, Phase 1; 0.19 px saturation, Phase 2). This makes
the filter lean on its prediction in proportion to how displaced we know the measurement to be —
which is *not* the same as rejecting it. A bounded bias is still information; an impulse outlier
is not, and the gate handles that case separately.

### 6.2 The gate can lock itself out — a measured defect

On a dim moving target the filter lost the track on **3 runs in 12**, and the mechanism was not
what it appeared. The measurements were good throughout (0.1–1.0 px error) and *every one* was
rejected from the fourth frame onward:

1. Initiating from a single detection starts velocity at **zero**.
2. The target moved ~100 px/s, i.e. 3.33 px/frame, so the prediction fell behind at once.
3. After a few accepted updates the covariance collapsed — confident position, and confident in a
   velocity that was wrong.
4. A 3.33 px innovation against that gate is rejected; a rejected measurement produces no update;
   so the velocity is never learned and the lag grows without bound.

Track error grew linearly to 27 px while the detector was working perfectly. **The failure is
silent** — nothing looks wrong except the output, which is what makes it dangerous.

`src/filtering/track.py` breaks that chain in three places:

| mechanism | which link it breaks |
|---|---|
| **Two-point initiation** — velocity from the first two detections | (1): the prediction is never systematically behind |
| **Ungated initiation** — no gate during confirmation | (4): gating against an unestablished state is the lockout |
| **Lockout detection** — N consecutive rejections re-initiate the track | (4): a track rejecting everything is evidence about the *track* |

Measured end to end through the real vision pipeline: track loss **3/12 → 0/12**, with median
error 0.17–0.41 px on every seed. Note that a gate rejecting every measurement is information
about the track, not about the measurements — treating it as the latter is what allowed the
silent divergence.

---

## 7. Control loop and PTZ kinematics

### 7.1 Pixel-to-angle mapping

```
deg_per_pixel = FOV_degrees / resolution_pixels
              = 4.0 / 640 = 0.00625 deg/px   (horizontal)
              = 3.0 / 480 = 0.00625 deg/px   (vertical)

angular_error_deg = pixel_error * deg_per_pixel
```

### 7.2 Controller

PID on angular error, with anti-windup and slew-rate limiting:

```
omega_cmd = clip(Kp*e + Ki*integral(e) + Kd*de/dt, -omega_max, +omega_max)
```

with `omega_max` = 5 °/s default (range 5–10 °/s). A two-loop structure — outer position/angle
loop feeding an inner rate loop — is the standard actively-stabilised camera design.

**Add Kalman-velocity feedforward.** Feedback-only PID always exhibits steady-state phase lag
against a continuously moving target; feedforward cancels it by commanding the camera to match
target velocity, not merely correct position error.

**The controller input is the Kalman-smoothed state estimate, never the raw per-frame centroid.**
This is a hard rule, not a preference (Phase 4). Camera jitter is specified at up to ±20 px/frame,
zero-mean, changing every frame — i.e. its energy sits at the Nyquist edge of a 30 Hz loop, well
above the closed-loop bandwidth. It is therefore **unobservable and uncontrollable**: no causal
controller can reject a disturbance that decorrelates between samples. Feeding raw centroids to
the PID does not attenuate jitter, it injects it — the derivative term differentiates it into
large spurious rate commands, the integral term accumulates its excursions, and the result is
actuator chatter that *increases* pointing error and drives lock/unlock oscillation.

The correct decomposition is:
- **Jitter** → absorbed by the Kalman filter as measurement noise. Left alone by the controller.
  It appears in the centroiding-error budget, not the pointing-error budget.
- **Platform motion** → a low-frequency bias inside the loop bandwidth. This is what the
  controller (specifically the integral term) exists to reject.

Practical consequences: the PID acts on the smoothed position/velocity estimate; the derivative
term is taken from the Kalman velocity state rather than by differencing measurements; and
`R` (section 6) must reflect the jitter magnitude so the filter smooths it by the right amount.

### 7.3 The binding feasibility constraint

```
omega_max = 5 deg/s
          = 5 / 0.00625 = 800 px/s
          = 26.7 px/frame at 30 Hz
```

At 10 °/s this is ~53 px/frame. **A target whose apparent motion exceeds this cannot be kept
centred.** Quantify the trackable velocity envelope (max target angular rate that keeps
steady-state pointing error ≤10 px given servo lag) and report it honestly. Teams that
characterise their own limits do better in technical Q&A than teams that quietly test only slow
targets.

### 7.4 Acquisition → tracking state machine

```
SEARCH  --(lock criterion held K frames)-->  TRACK
TRACK   --(N consecutive missed detections)-->  COAST
COAST   --(re-lock)-->  TRACK
COAST   --(coast timeout)-->  SEARCH
```

- **SEARCH:** open-loop Archimedean spiral / raster scan over the uncertainty region.
  **Arm spacing is FOV-derived, not a stored constant:**
  `arm_spacing = coverage_factor * min(fov_width_px, fov_height_px)` with `coverage_factor ≈ 0.9`
  for overlap margin. Using the *limiting* FOV dimension guarantees the sweep leaves no gaps;
  using a hardcoded value (the old `arm_spacing_px: 300`) either wastes time or silently opens
  coverage holes the moment the FOV or resolution changes.
- **TRACK:** closed-loop PID + feedforward.
- **COAST:** Kalman prediction only, then a small local spiral around the predicted position to
  meet the ≤1 s re-acquisition requirement.
- Use **hysteresis** on the lock criterion (different enter/exit thresholds) to prevent mode
  chatter at marginal SNR.

### 7.5 Acquisition time must be logged as two populations

The state machine must record, at frame 0, whether the beacon was inside the initial viewport,
and tag the resulting acquisition event accordingly:

- **In-FOV acquisition** — beacon already in the initial viewport. Time is detection-limited
  (essentially `K` frames plus pipeline latency). The spec's ≤2 s budget is comfortably
  achievable here, and this is the number the demo should lead with.
- **Search-limited acquisition** — beacon outside the initial viewport. Time is dominated by the
  SEARCH sweep and is bounded below by the slew ceiling. For the default configuration:

```
uncertainty area / arm spacing = 2000*2000 / (0.9 * 480) ≈ 9.3e3 px of path
at 5 deg/s = 800 px/s          -> ≈ 11.6 s worst case
at 10 deg/s = 1600 px/s        -> ≈ 5.8 s worst case
```

**This exceeds the ≤2 s requirement by design, and no amount of algorithm tuning changes it** —
it is set by canvas area, FOV and the mechanical slew limit alone. `src/config.py` computes this
bound for the active configuration and emits a startup warning when it exceeds the budget, so the
number is visible from the first run rather than discovered late.

Report both populations separately with counts, mean and worst case, and state the arithmetic
above in the report. This is the same honesty posture as the trackable-velocity envelope in
section 7.3: characterising our own limits scores better in technical Q&A than a pooled headline
number that silently depends on where the beacon happened to start. Pooling the two also makes
the metric non-reproducible, since `target.initial_position: "random"` would turn the headline
acquisition time into an initial-condition lottery.

---

### 7.6 Measured control results (Phase 4)

**The PID gains are validated and the PROVISIONAL marker is lifted.** The gate was a step-response
test run **with the real Kalman filter in the loop**, because the controller consumes smoothed
state and the filter's lag therefore sits inside the loop. That matters: the analytic sizing
(Kp = 8) assumed pointing error maps straight to a rate command, and measured against a perfect
measurement it settles a 100 px step in 0.30 s — but with the filter in the loop it takes 1.07 s.
Kp = 12 restores 0.40 s. A gain validated against an idealised measurement is not validated
against the system we ship, the same lesson as the R calibration.

Final gains, each sized by the measurement that governs it:

| gain | value | sized by |
|---|---|---|
| `kp` | 12.0 | step response with the filter in the loop: 0.40 s settle vs 1.07 s at Kp=8 |
| `ki` | 1.0 | platform-drift rejection at 150 px/s: 19.5 px at ki=0, 9.2 px at 0.5, 4.3 px at 1.0 |
| `kd` | 0.0 | measured as zero — see below |

**`kd = 0` is a measured result, not an omission.** With Kalman-velocity feedforward already
cancelling target motion, the derivative contributes no useful damping and amplifies estimator
noise: `kd=1.5` costs 2.58 px static error against 0.75 px, and settles in 0.87 s against 0.40 s.

**A derivative double-count bug was found on the way there.** An earlier version passed the
*target velocity* as the derivative input. The derivative must act on the rate of change of the
**error**, which is target velocity minus camera rate; feeding the target velocity alone makes
`kd` a second feedforward with the wrong gain, counting the target's motion twice. It cost a
factor of 2–4 in steady-state pointing error on a moving target (8.44 px against 3.62 px at
100 px/s) and presented as "kd is harmful" rather than as a double-count.

**Feedforward and the integral are partially redundant, and the test must account for it.** On a
*constant-velocity* target the integral reaches the same place by a different route — it
integrates a constant error into a constant rate — so with both active, disabling feedforward
appears to *help* (4.5 px against 6.7 px). Isolating with `ki = 0` shows its real contribution
(6.7 px against 10.0 px). Feedforward earns its place on *manoeuvring* targets, where it responds
to a velocity change immediately while the integral must wind up again.

### 7.7 The trackable-velocity envelope, including where it fails

`docs/figures/trackable_velocity.png`. Steady-state pointing error against target speed, swept
past the point of failure rather than stopping at the last speed that works.

| target speed | steady-state pointing error |
|---|---|
| 100 px/s | 3.7 px |
| 200 px/s | 6.9 px |
| **280 px/s** | **10.0 px — budget crossed** |
| 400 px/s | 13.9 px |
| 600 px/s | 20.2 px |
| 800 px/s | 220 px — slew ceiling |
| 1200 px/s | 634 px |

**The headline is that the loop bandwidth binds long before the mechanism does.** The slew ceiling
is 800 px/s (26.7 px/frame), but the 10 px pointing budget is exceeded at **279 px/s — 35% of the
mechanical ceiling**, i.e. 1.75 deg/s against the 5 deg/s the gimbal can actually slew. Quoting
the slew ceiling as "the" limit would overstate our envelope by a factor of about three.

Beyond 800 px/s the error jumps by an order of magnitude: past the ceiling the camera cannot keep
up regardless of gains, and no controller change alters that.

**The FOV trade-off does not run the way the usual argument suggests.** The standard reasoning is
that a wider FOV helps acquisition while hurting fast tracking in pixel terms. The second half
holds. The first half does **not**, and the reason is structural: our uncertainty region is a
*pixel* canvas and the viewport is a fixed *pixel* count, so a wider FOV does not let the camera
see more of the canvas. Spiral arm spacing stays `0.9 × min(fov_px)` = 432 px whatever the FOV, so
the path length is unchanged, while the angular scale coarsens and a fixed angular slew rate
sweeps *fewer* pixels per second:

| FOV | deg/px | canvas coverage time | pixel slew ceiling |
|---|---|---|---|
| 2° | 0.00313 | **5.8 s** | 1600 px/s |
| 4° | 0.00625 | 11.6 s | 800 px/s |
| 8° | 0.01250 | **23.1 s** | 400 px/s |

So in this formulation a wider FOV is worse on **both** counts. The usual argument applies to a
system whose uncertainty region is angular and whose detector subtends the FOV; the specification
fixes canvas (parameter 1) and resolution (parameter 3) independently of FOV (parameter 4), which
decouples them. The intuition is natural and wrong here, so it is worth stating explicitly in the
report rather than leaving a reviewer to assume the usual trade-off applies.

The tracking effect is sharp rather than gradual, which is itself evidence the ceiling is the
mechanism: at 300 px/s the 8° FOV is marginally *better* (10.07 px against 10.74 px), and at
400 px/s — exactly its ceiling — its error jumps to 110 px against 13.9 px.

---

## 8. Dual-mode input architecture

`FrameSource` protocol (see `src/framesource.py`):

```python
get_frame() -> FrameData(frame, timestamp, frame_index, ground_truth | None)
```

**Mode A — `SimulationFrameSource`:** renders the world, applies the controller's pan/tilt,
extracts the 640×480 viewport, applies noise/atmosphere, returns the frame plus the known
sub-pixel ground-truth centroid.

**Mode B — `VideoFrameSource`:** reads the evaluator's `.mp4` at native rate, returns the frame
with `ground_truth=None` (or loaded from a sidecar file if evaluators supply one). The PTZ loop
is bypassed — the video *is* the scene, and we report instantaneous centroiding/pointing error
directly.

**Mode B robustness requirements** (this is 30% of the grade on inputs we cannot see):
- Auto-detect resolution; do not assume 640×480 or 2000×2000.
- Handle colour input by converting to grayscale (spec says monochrome, evaluator video may not be).
- Normalise intensity per frame; never assume a brightness range.
- Every threshold derived from frame statistics.
- Emit centroiding-error logs in a machine-checkable format regardless of whether ground truth is
  available (when absent, log the estimated centroid trajectory and derived stability metrics).
  Coordinates are reported in **full-frame source pixel coordinates** under the pixel-centre
  convention (see `CLAUDE.md` → Coordinate conventions), stated in the log header so evaluator
  comparison needs no coordinate negotiation.

**Lossy compression is part of the input, not an implementation detail.** Evaluator input is
`.mp4` at 30 fps, so it has been through H.264. That is not a neutral transport:

- An 8×8/4×4 transform smears a small bright spot across block boundaries and adds ringing around
  the highest-contrast feature in the frame — which *is* our beacon.
- Impulse noise (salt & pepper, up to 10% per spec) is close to worst-case content for a DCT
  codec, so what reaches us is not the noise model we simulated but its compressed residue.
- Chroma subsampling and any studio/full range mismatch shift the luma histogram, which moves
  every statistically-derived threshold.

**Therefore our own Phase 6 test videos must round-trip through real H.264 at a realistic
bitrate** — never raw or lossless frames. Validating against lossless video validates nothing we
will actually be scored on. We additionally measure the **centroid bias attributable to
compression alone** by running the identical frame sequence through the pipeline lossless vs
encoded and differencing the results; that delta is a report figure and bounds the error floor
Benchmark-2 can possibly achieve.

**Throughput risk specific to Mode B:** the spec says the video covers "a complete screen", which
may mean the full 2000×2000 canvas rather than a 640×480 viewport. That changes the budget
twice over — decode cost for 4 Mpx frames at 30 fps, and full-frame acquisition search over 4
Mpx before lock is established. ROI processing only rescues us *after* lock. Measure decode and
first-lock cost on full-canvas video early in Phase 6; if it does not hold ≥20 FPS, the fix is a
coarse-to-fine search (downsampled full-frame scan to localise, full-resolution ROI to centroid),
not a threshold hack.

---

## 9. Telemetry and metrics

Per-frame CSV/JSON record:

```
timestamp, frame_index, mode, gt_x, gt_y, est_x, est_y,
centroid_error_px, pointing_error_px, locked, state,
detection_snr, processing_ms, fps_instant,
camera_pan_deg, camera_tilt_deg
```

Summary report (HTML/PDF, auto-generated):
- Simulation duration, total frames
- Acquisition time, all re-acquisition events and times
- Mean / max / RMSE centroiding error; mean / max pointing error
- Lock retention rate, target loss rate
- Mean / min / max FPS; mean processing time per frame
- **Pipeline capacity (unthrottled)** — see below; distinct from the real-time rate
- Configuration snapshot (so results are reproducible)
- The metric definitions used (state them in the header — removes evaluator ambiguity)

---

### 9.1 Unthrottled benchmark mode (capacity vs. rate)

The three clocks must be measured, not inferred. In a real-time run the simulator produces frames
at 30 Hz, so a vision pipeline capable of 200 Hz still logs 30 FPS. That understates us against
the ≥20 FPS requirement and, worse, **hides regressions**: throughput could silently degrade from
200 Hz to 31 Hz and every log would still read 30 FPS, right up until the frame we miss.

So telemetry provides two separate figures, both in the summary report:

- **Real-time rate** — frames actually delivered and processed per wall-clock second in a normal
  run. This is what the GUI shows and what the live demo is judged on.
- **Pipeline capacity** — an unthrottled mode that feeds a pre-generated frame buffer through the
  identical vision + filtering + control chain as fast as it will go, with rendering, noise
  synthesis, GUI and disk I/O excluded from the timed region. Report mean/p50/p95/max per-frame
  processing time and the derived max sustainable FPS.

Capacity is what proves the ≥20 FPS claim with headroom; run it as a CI-style check so a
throughput regression is caught the day it lands rather than during the demo. Report both, and
say which is which — conflating them is exactly the ambiguity we are otherwise being careful to
avoid.

---

## 10. Performance engineering

- **ROI-limited processing** around the Kalman prediction once locked (default 64×64). This alone
  is the difference between real-time and not.
- **Vectorise** all noise and centroid math in NumPy; JIT only the residual Python loops with
  Numba `@njit(nogil=True)`.
- **Thread decoupling:** producer (render + noise) → bounded queue → consumer (vision + control)
  → Qt signals → GUI. NumPy and OpenCV release the GIL during array operations, so real
  parallelism is achievable for the numeric hotspots.
- **Memory:** allocate the 2000×2000 canvas once and reuse buffers; prefer `uint8`; avoid
  per-frame reallocation.
- **Profile with** `cProfile` and `line_profiler`; log live FPS so regressions are visible
  immediately.

---

## 11. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Overfitting to own synthetic noise; fails Benchmark-2 | High | High | Parameter-free thresholding, per-frame normalisation, test on varied unseen-style video |
| Full-canvas processing kills FPS | Medium | High | ROI-around-prediction; profile from day one |
| Fast targets exceed slew rate | Medium | Medium | Quantify trackable envelope; Kalman feedforward; document the limit |
| Ambiguous acquisition-clock definition costs benchmark marks | Medium | High | Define precisely, state definition in log headers |
| PyInstaller misses Numba/Qt/OpenCV binaries | Medium | Medium | Build at 50% completion, test on clean VM, add hidden-imports |
| Salt & pepper false positives break lock | Medium | Medium | Median prefilter + blob-size gate + Mahalanobis gate |
| Mode B resolution/colour mismatch | Medium | Medium | Auto-detect, normalise coordinates, colour→gray conversion |
| H.264 artifacts break thresholds tuned on lossless frames | High | High | All Phase 6 test video round-trips through real H.264 at realistic bitrate; measure compression-only centroid bias (section 8) |
| Full-canvas (2000×2000) Mode B video busts the FPS budget on decode + pre-lock search | Medium | High | Measure decode and first-lock cost early in Phase 6; coarse-to-fine search if needed (section 8) |
| Absolute-pixel vision parameters fail on unseen resolution/spot size | High | High | Scale-relative parameterisation from runtime spot-scale estimate; absolutes are fallbacks only |
| Search-limited acquisition exceeds the ≤2 s budget | Certain (by arithmetic) | Medium | Not fixable by tuning; compute the bound at startup, log the two acquisition populations separately, report honestly (section 7.5) |
| Fixed Kalman R mis-scales the validation gate at SNR extremes | Medium | High | Derive R per frame from detection SNR; validate with NIS consistency check (section 6) |
| Raw centroids fed to PID amplify jitter into actuator chatter | Medium | High | Controller consumes Kalman-smoothed state only; derivative from filter velocity state (section 7.2) |
| Throughput regression hidden behind the 30 Hz real-time cap | Medium | Medium | Unthrottled capacity benchmark reported separately from real-time rate (section 9.1) |

---

## 12. Caveats to state honestly in the report

- Real on-orbit acquisition times (SILEX up to ~130 s, LCRD ~30–45 s) are far longer than the
  spec's ≤2 s. The spec target is valid as a simulation convenience, not as a claim about real
  FSOC systems.
- "SNR" is defined inconsistently across the centroiding literature (peak SNR vs. total-photon
  SNR vs. ROI SNR). We settle it explicitly in `CLAUDE.md` -> Metric definitions: `snr_aperture`
  is primary and is the SNR-curve x-axis, `snr_peak` is logged alongside, and both use robust
  median/MAD background statistics over an annulus. The scaling laws remain the defensible
  statements; specific numeric accuracies are regime indicators.
- Deep-learning FPS figures depend heavily on hardware and optimisation. Validate any AI
  component's throughput on the actual demo machine, never assume.
- Patent-sourced coarse/fine accuracy figures illustrate the general PAT hierarchy; specific
  missions vary considerably.
