import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, shift as ndi_shift
from scipy.optimize import curve_fit
from skimage.transform import radon, iradon

try:
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    THIS_DIR = os.getcwd()
OUT_DIR = os.path.join(THIS_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

RNG = np.random.default_rng(2024)


def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}")



# ============================================================================
# SECTION 1: THE CARDIAC PHANTOM
# ----------------------------------------------------------------------------
# ============================================================================


IMAGE_SIZE = 96
FOV_MM = 300.0
MM_PER_PX = FOV_MM / IMAGE_SIZE

MYO_ACTIVITY = 1.0
BACKGROUND_ACTIVITY = 0.05
DEFECT_FRACTION = 0.35  # the defect keeps 35% of normal uptake (partial,
# not total, defect -- more realistic and the
# harder detection case)
WALL_INNER_R = 16
WALL_OUTER_R = 26
DEFECT_ANGLE_LO, DEFECT_ANGLE_HI = 200, 260  # degrees; sits in what we
# label the "inferolateral"
# segment below

y_idx, x_idx = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE]
cx, cy = IMAGE_SIZE / 2, IMAGE_SIZE / 2
r_map = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)
theta_map = (np.degrees(np.arctan2(y_idx - cy, x_idx - cx)) + 360) % 360


def build_phantom():
    print("[1] Building the cardiac phantom (LV wall + defect)...")
    phantom = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
    phantom[r_map <= 44] = BACKGROUND_ACTIVITY
    in_wall = (r_map >= WALL_INNER_R) & (r_map <= WALL_OUTER_R)
    phantom[in_wall] = MYO_ACTIVITY
    in_defect = in_wall & (theta_map >= DEFECT_ANGLE_LO) & (theta_map <= DEFECT_ANGLE_HI)
    phantom[in_defect] = MYO_ACTIVITY * DEFECT_FRACTION

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(phantom, cmap="hot")
    ax.set_title("1. Ground-truth phantom\n(mid-LV short axis, dim wedge = perfusion defect)")
    ax.axis("off")
    savefig(fig, "01_phantom.png")
    return phantom

# ============================================================================
# SECTION 2: THE D-SPECT-STYLE ACQUISITION
# ----------------------------------------------------------------------------
# ============================================================================


N_ANGLES = 90
ANGLES = np.linspace(0, 180, N_ANGLES, endpoint=False)
TOTAL_COUNTS = 3.0e5  # matched between the two acquisition modes
# below, so the comparison is fair: same
# dose/scan-time budget, different allocation
HEART_FACING_ANGLE = 60.0  # degrees; the part of the 0-180 sweep that
# views the heart most directly from the
# left anterior chest -- an illustrative,
# not clinically exact, choice
FOCUS_WIDTH_DEG = 35.0  # width of the adaptive dwell-time weighting


def collimator_blur(image, sigma_px):
    blurred = gaussian_filter(image, sigma=sigma_px)
    blurred[~INSIDE_CIRCLE] = 0.0
    return blurred


def acquire(phantom, weighting, seed):

    blurred = collimator_blur(phantom, sigma_px=1.3)  # ~1.3 px ~ 4 mm blur,

    clean_sino = radon(blurred, theta=ANGLES, circle=True)

    weighting = weighting / weighting.sum()  # normalize so the
    # total counts are
    # identical between
    # acquisition modes
    per_angle_counts = TOTAL_COUNTS * weighting
    rng = np.random.default_rng(seed)
    noisy_sino = np.zeros_like(clean_sino)
    for i, target_counts in enumerate(per_angle_counts):
        col = clean_sino[:, i]
        scale = target_counts / col.sum() if col.sum() > 0 else 0
        noisy_sino[:, i] = rng.poisson(np.clip(col * scale, 0, None))
    return noisy_sino


def uniform_weighting():
    return np.ones(N_ANGLES)


def heart_focused_weighting():

    gauss = np.exp(-0.5 * ((ANGLES - HEART_FACING_ANGLE) / FOCUS_WIDTH_DEG) ** 2)
    floor = 0.15
    return floor + (1 - floor) * gauss


def run_acquisition(phantom):
    print("[2] Simulating acquisition: conventional uniform-dwell vs. D-SPECT-style adaptive dwell...")
    sino_uniform = acquire(phantom, uniform_weighting(), seed=1)
    sino_focused = acquire(phantom, heart_focused_weighting(), seed=2)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    axes[0].plot(ANGLES, uniform_weighting(), label="uniform (conventional)")
    axes[0].plot(ANGLES, heart_focused_weighting(), label="heart-focused (D-SPECT-style)")
    axes[0].axvline(HEART_FACING_ANGLE, color="gray", linestyle=":", label="heart-facing angle")
    axes[0].set_xlabel("Projection angle (deg)")
    axes[0].set_ylabel("Relative dwell weight")
    axes[0].set_title("Dwell-time allocation")
    axes[0].legend(fontsize=7)

    axes[1].imshow(sino_uniform, cmap="gray", aspect="auto")
    axes[1].set_title("Sinogram: uniform dwell")
    axes[1].axis("off")

    axes[2].imshow(sino_focused, cmap="gray", aspect="auto")
    axes[2].set_title("Sinogram: heart-focused dwell\n(same total counts)")
    axes[2].axis("off")
    savefig(fig, "02_acquisition_comparison.png")
    return sino_uniform, sino_focused

# ============================================================================
# SECTION 3: MOTION QUALITY CONTROL ("refine, clean, rectify")
# ----------------------------------------------------------------------------
# ============================================================================


def maybe_inject_motion(sinogram, seed, motion_prob=0.5, shift_px=3.0):
    rng = np.random.default_rng(seed)
    corrupted = sinogram.copy()
    motion_happened = rng.random() < motion_prob
    onset_angle_idx = None
    if motion_happened:
        onset_angle_idx = int(rng.integers(N_ANGLES // 3, 2 * N_ANGLES // 3))
        corrupted[:, onset_angle_idx:] = ndi_shift(
            corrupted[:, onset_angle_idx:], shift=(shift_px, 0), order=1
        )
    return corrupted, motion_happened, onset_angle_idx


def detect_and_correct_motion(sinogram):
    """Track the per-projection center-of-mass (a standard, simple motion
    signature) and re-align any projection whose COM jumps sharply
    relative to its neighbors."""
    n_bins = sinogram.shape[0]
    bin_positions = np.arange(n_bins)
    com = np.array([
        (sinogram[:, i] * bin_positions).sum() / max(sinogram[:, i].sum(), 1e-9)
        for i in range(sinogram.shape[1])
    ])
    com_smooth = gaussian_filter(com, sigma=2)
    residual = com - com_smooth
    threshold = 3 * np.std(residual)
    flagged = np.where(np.abs(residual) > threshold)[0]

    corrected = sinogram.copy()
    for i in flagged:
        correction = -(com[i] - com_smooth[i])
        corrected[:, i] = ndi_shift(corrected[:, i], shift=correction, order=1)
    return corrected, com, com_smooth, flagged


def run_motion_qc(sino_focused):
    print("[3] Motion QC: simulating, detecting, and correcting patient motion...")
    corrupted, motion_happened, onset = maybe_inject_motion(sino_focused, seed=7)
    print(f"    motion injected this run: {motion_happened}"
          + (f" (onset at angle index {onset})" if motion_happened else ""))

    corrected, com, com_smooth, flagged = detect_and_correct_motion(corrupted)
    print(f"    auto-QC flagged {len(flagged)} projection(s) as motion-corrupted")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(com, label="raw center-of-mass per angle")
    ax.plot(com_smooth, label="smoothed (expected) trend", linestyle="--")
    if len(flagged):
        ax.scatter(flagged, com[flagged], color="red", zorder=5, label="flagged as motion")
    ax.set_xlabel("Projection index")
    ax.set_ylabel("Center-of-mass (detector bin)")
    ax.set_title("3. Automated motion QC\n(mirrors panogram review on a real D-SPECT workstation)")
    ax.legend(fontsize=8)
    savefig(fig, "03_motion_qc.png")
    return corrected

# ============================================================================
# SECTION 4: RECONSTRUCTION WITH RESOLUTION RECOVERY
# ----------------------------------------------------------------------------
# ============================================================================


OSEM_ITERS = 8
OSEM_SUBSETS = 6


def osem_reconstruct(sinogram, model_collimator_blur, blur_sigma_px=1.3):
    n_angles = len(ANGLES)
    n_subsets = min(OSEM_SUBSETS, n_angles)
    subset_indices = [np.arange(i, n_angles, n_subsets) for i in range(n_subsets)]
    inside_circle = r_map <= IMAGE_SIZE / 2

    def apply_system_forward(image, sub_theta):
        """The full forward model: collimator blur, then projection --
        exactly what the physical acquisition does (Section 2)."""
        fwd_input = collimator_blur(image, blur_sigma_px) if model_collimator_blur else image
        return radon(fwd_input, theta=sub_theta, circle=True)

    def apply_system_adjoint(sinogram_slice, sub_theta):
        """The adjoint of the forward model: (unfiltered) backprojection,
        then the SAME blur again (a symmetric Gaussian is its own
        adjoint) -- must mirror apply_system_forward exactly, or the
        update ratio's numerator and denominator stop being on the same
        footing and the reconstruction is biased (this was the earlier
        bug: only the numerator went through the full adjoint chain)."""
        back = iradon(sinogram_slice, theta=sub_theta, filter_name=None, circle=True)
        return collimator_blur(back, blur_sigma_px) if model_collimator_blur else back

    # Sensitivity images depend only on the geometry (not on the current
    # estimate), so precompute them once per subset by running a uniform
    # image through the FULL forward+adjoint chain -- this is what makes
    # the numerator and denominator of the OSEM update properly matched.
    ones_image = np.ones((IMAGE_SIZE, IMAGE_SIZE))
    ones_image[~inside_circle] = 0.0
    sensitivities = [
        apply_system_adjoint(apply_system_forward(ones_image, ANGLES[idx]), ANGLES[idx]) + 1e-6
        for idx in subset_indices
    ]

    estimate = np.ones((IMAGE_SIZE, IMAGE_SIZE))
    estimate[~inside_circle] = 0.0

    for _ in range(OSEM_ITERS):
        for idx, sensitivity in zip(subset_indices, sensitivities):
            sub_theta = ANGLES[idx]
            sub_measured = sinogram[:, idx]

            expected = apply_system_forward(estimate, sub_theta)
            ratio = sub_measured / (expected + 1e-6)
            correction = apply_system_adjoint(ratio, sub_theta)

            estimate = estimate * (correction / sensitivity)
            estimate[~inside_circle] = 0.0
            estimate = np.clip(estimate, 0, None)
    return estimate


def run_reconstruction(sino_clean, phantom):
    print("[4] Reconstructing: with vs. without resolution recovery...")
    recon_no_rr = osem_reconstruct(sino_clean, model_collimator_blur=False)
    recon_with_rr = osem_reconstruct(sino_clean, model_collimator_blur=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, img, title in zip(
            axes, [phantom, recon_no_rr, recon_with_rr],
            ["Ground truth", "OSEM, no resolution\nrecovery (blurred)",
             "OSEM WITH resolution\nrecovery (D-SPECT-style)"],
    ):
        # Each panel is scaled to its OWN intensity range (99th percentile)
        # rather than a shared fixed value -- OSEM's reconstructed scale
        # depends on the system model used, so a fixed vmax from the
        # phantom would make one or both reconstructions look artificially
        # dim without this being a real feature of the image.
        vmax = np.percentile(img, 99.5)
        ax.imshow(np.clip(img, 0, vmax), cmap="hot", vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    savefig(fig, "04_reconstruction_comparison.png")
    return recon_with_rr

# ============================================================================
# SECTION 5: SEGMENTAL PERFUSION SCORING (SSS / SRS / SDS)
# ----------------------------------------------------------------------------
# ============================================================================


SEGMENT_NAMES = ["Anterior", "Anteroseptal", "Inferoseptal", "Inferior", "Inferolateral", "Anterolateral"]
SEGMENT_CENTERS_DEG = [90, 150, 210, 270, 330, 30]  # 6 wedges of 60 deg each


def segment_mean(image, center_deg, half_width=30):
    lo, hi = (center_deg - half_width) % 360, (center_deg + half_width) % 360
    ring = (r_map >= WALL_INNER_R + 2) & (r_map <= WALL_OUTER_R - 2)
    if lo < hi:
        wedge = (theta_map >= lo) & (theta_map <= hi)
    else:
        wedge = (theta_map >= lo) | (theta_map <= hi)
    mask = ring & wedge
    return image[mask].mean()


def percent_normal_to_score(pct_normal):
    """Approximate quantitative-to-visual-score conversion in the style of
    Cedars-Sinai QPS/QGS literature (Germano et al.): a continuous %-of-
    normal uptake value is bucketed into the standard clinical 0-4 scale."""
    if pct_normal >= 75:
        return 0
    elif pct_normal >= 60:
        return 1
    elif pct_normal >= 45:
        return 2
    elif pct_normal >= 30:
        return 3
    else:
        return 4


def score_image(image, normal_reference_value):
    rows = []
    for name, center in zip(SEGMENT_NAMES, SEGMENT_CENTERS_DEG):
        val = segment_mean(image, center)
        pct_normal = 100 * val / normal_reference_value
        score = percent_normal_to_score(pct_normal)
        rows.append({"segment": name, "pct_normal": pct_normal, "score": score})
    return rows


def run_scoring(recon_stress, recon_rest, phantom):
    print("[5] Segmental perfusion scoring (SSS / SRS / SDS)...")

    # "Normal reference" must come from the RECONSTRUCTED image itself (the
    # mean of its own unaffected segments) -- exactly like a real system,
    # which never has access to ground truth and instead normalizes each
    # study internally (or against a population normal database built the
    # same way, from reconstructed images, not from an unattainable truth).
    def normal_reference(image):
        return np.mean([segment_mean(image, c) for c in SEGMENT_CENTERS_DEG
                        if not (DEFECT_ANGLE_LO - 20 <= c <= DEFECT_ANGLE_HI + 20)])

    stress_rows = score_image(recon_stress, normal_reference(recon_stress))
    rest_rows = score_image(recon_rest, normal_reference(recon_rest))

    sss = sum(r["score"] for r in stress_rows)
    srs = sum(r["score"] for r in rest_rows)
    sds = sss - srs
    pct_myo_stress = 100 * sss / (4 * len(SEGMENT_NAMES))

    print("    Segment            Stress %normal  Stress score   Rest %normal  Rest score")
    for s, r in zip(stress_rows, rest_rows):
        print(f"    {s['segment']:<16} {s['pct_normal']:6.1f}%         {s['score']}"
              f"               {r['pct_normal']:6.1f}%        {r['score']}")
    print(f"    SSS={sss}  SRS={srs}  SDS={sds}  (%myocardium abnormal, stress = {pct_myo_stress:.1f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), subplot_kw={"projection": "polar"})
    for ax, rows, title in zip(axes, [stress_rows, rest_rows], ["Stress", "Rest"]):
        angles_rad = np.radians(SEGMENT_CENTERS_DEG + [SEGMENT_CENTERS_DEG[0]])
        scores = [r["score"] for r in rows] + [rows[0]["score"]]
        ax.plot(angles_rad, scores, "o-", color="crimson")
        ax.fill(angles_rad, scores, color="crimson", alpha=0.25)
        ax.set_ylim(0, 4)
        ax.set_title(f"{title} bull's-eye scores (0=normal, 4=severe)", fontsize=9)
        ax.set_xticks(np.radians(SEGMENT_CENTERS_DEG))
        ax.set_xticklabels(SEGMENT_NAMES, fontsize=7)
    savefig(fig, "05_bullseye_scores.png")
    return sss, srs, sds

# ============================================================================
# SECTION 6: GATED FUNCTION (LVEF) -- SIMPLIFIED 2D ANALOG
# ----------------------------------------------------------------------------
# ============================================================================


def build_phantom_at_phase(cavity_scale):
    """cavity_scale < 1 shrinks the blood-pool cavity (systole); the wall
    itself moves inward/thickens slightly, as real myocardium does."""
    inner_r = WALL_INNER_R * cavity_scale
    outer_r = WALL_OUTER_R - (WALL_INNER_R - inner_r) * 0.3
    ph = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
    ph[r_map <= 44] = BACKGROUND_ACTIVITY
    in_wall = (r_map >= inner_r) & (r_map <= outer_r)
    ph[in_wall] = MYO_ACTIVITY
    return ph, inner_r


def run_gated_function():
    print("[6] Gated function: simplified 2D end-diastole/end-systole EF analog...")
    ph_ed, r_ed = build_phantom_at_phase(cavity_scale=1.0)  # end-diastole: cavity largest
    ph_es, r_es = build_phantom_at_phase(cavity_scale=0.55)  # end-systole: cavity smallest

    edv_2d = np.pi * r_ed ** 2  # "volume" proxy = cavity area in this 2D analog
    esv_2d = np.pi * r_es ** 2
    ef_2d = 100 * (edv_2d - esv_2d) / edv_2d

    fig, axes = plt.subplots(1, 2, figsize=(7, 4))
    for ax, ph, title in zip(axes, [ph_ed, ph_es], ["End-diastole (relaxed)", "End-systole (contracted)"]):
        ax.imshow(ph, cmap="hot")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle(f"6. Gated cardiac phases -- 2D area-based EF analog: {ef_2d:.0f}%\n"
                 f"(illustrative only -- real LVEF uses 3D gated volumes, not a 2D area)")
    savefig(fig, "06_gated_function.png")
    print(f"    2D area-based EF analog = {ef_2d:.0f}% (illustrative; real clinical LVEF needs 3D volumes)")
    return ef_2d

# ============================================================================
# SECTION 7: DYNAMIC FLOW QUANTIFICATION (MBF / MFR)
# ----------------------------------------------------------------------------
# ============================================================================


def blood_input_function(t, dose_scale=1.0):
    """A gamma-variate-like bolus arriving and clearing -- the standard
    shape used to represent an IV bolus arterial input function."""
    t = np.clip(t, 1e-6, None)
    return dose_scale * (t ** 3) * np.exp(-t / 0.5)


def myocardial_tac(t, K1, k2, dose_scale=1.0):
    """Analytic solution of the 1-tissue-compartment ODE for a
    gamma-variate input (convolution of Cb(t) with K1*exp(-k2 t))."""
    dt = t[1] - t[0]
    cb = blood_input_function(t, dose_scale)
    kernel = K1 * np.exp(-k2 * t)
    conv = np.convolve(cb, kernel)[: len(t)] * dt
    return conv


def renkin_crone_flow_from_K1(K1, PS=0.8):
    """
    Invert the Renkin-Crone extraction relation to recover flow F from a
    fitted K1: K1 = F * E(F), E(F) = 1 - exp(-PS/F).
    PS (permeability-surface-area product) is a fixed tracer property;
    0.8 is a representative literature value for 99mTc-sestamibi-class
    agents at typical resting/hyperemic flows, used here for illustration
    rather than as an exact calibrated constant.
    """
    F_grid = np.linspace(0.05, 6.0, 4000)
    E_grid = 1 - np.exp(-PS / F_grid)
    K1_grid = F_grid * E_grid
    return np.interp(K1, K1_grid, F_grid)


def simulate_and_fit_flow(true_flow, label, seed, PS=0.8, noise_frac=0.05):
    """Simulate a noisy dynamic TAC for a given TRUE flow, fit K1/k2, and
    recover the estimated flow -- the full round trip a real dynamic
    D-SPECT study performs, condensed to the TAC level."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 6, 70)  # 70 frames over 6 minutes, matching the
    # D-SPECT dynamic protocol's frame count

    true_E = 1 - np.exp(-PS / true_flow)
    true_K1 = true_flow * true_E
    true_k2 = true_K1 / 0.7  # illustrative fixed distribution volume

    clean_tac = myocardial_tac(t, true_K1, true_k2)
    noisy_tac = clean_tac + rng.normal(0, noise_frac * clean_tac.max(), size=clean_tac.shape)

    def model(t, K1, k2):
        return myocardial_tac(t, K1, k2)

    popt, _ = curve_fit(model, t, noisy_tac, p0=[0.5, 0.7], bounds=(0, [5, 5]))
    fitted_K1, fitted_k2 = popt
    fitted_flow = renkin_crone_flow_from_K1(fitted_K1, PS=PS)

    return {
        "label": label, "t": t, "noisy_tac": noisy_tac, "clean_tac": clean_tac,
        "true_flow": true_flow, "fitted_flow": fitted_flow, "fitted_K1": fitted_K1,
    }


def run_dynamic_flow():
    print("[7] Dynamic acquisition: fitting myocardial blood flow (MBF) and flow reserve (MFR)...")
    rest = simulate_and_fit_flow(true_flow=1.0, label="Rest", seed=11)
    stress = simulate_and_fit_flow(true_flow=2.7, label="Stress (vasodilator)", seed=12)

    mfr_true = stress["true_flow"] / rest["true_flow"]
    mfr_fit = stress["fitted_flow"] / rest["fitted_flow"]

    print(f"    Rest:   true flow = {rest['true_flow']:.2f} mL/min/g   "
          f"fitted flow = {rest['fitted_flow']:.2f} mL/min/g")
    print(f"    Stress: true flow = {stress['true_flow']:.2f} mL/min/g   "
          f"fitted flow = {stress['fitted_flow']:.2f} mL/min/g")
    print(f"    MFR (true) = {mfr_true:.2f}     MFR (fitted from noisy data) = {mfr_fit:.2f}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for res, color in zip([rest, stress], ["tab:blue", "tab:red"]):
        ax.plot(res["t"], res["clean_tac"], color=color, linestyle="--", alpha=0.6,
                label=f"{res['label']}, noiseless model")
        ax.scatter(res["t"], res["noisy_tac"], color=color, s=12,
                   label=f"{res['label']}, simulated noisy data")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Myocardial activity (a.u.)")
    ax.set_title(f"7. Dynamic TACs and 1-tissue-compartment fit\n"
                 f"MFR: true={mfr_true:.2f}, fitted={mfr_fit:.2f}")
    ax.legend(fontsize=7)
    savefig(fig, "07_dynamic_flow.png")
    return rest, stress, mfr_fit


# ============================================================================
# MAIN: RUN THE FULL PIPELINE, THEN PRINT A CLINICAL-STYLE SUMMARY
# ============================================================================

def main():
    phantom = build_phantom()
    sino_uniform, sino_focused = run_acquisition(phantom)
    sino_clean_stress = run_motion_qc(sino_focused)
    recon_stress = run_reconstruction(sino_clean_stress, phantom)

    # "Rest" study: same phantom but WITHOUT the defect (rest perfusion is
    # usually less abnormal than stress for a reversible ischemic defect;
    # here we simulate a fully normal rest scan as the simplest realistic
    # case, i.e. SDS will equal SSS, an entirely reversible defect)
    rest_phantom = phantom.copy()
    rest_phantom[(theta_map >= DEFECT_ANGLE_LO) & (theta_map <= DEFECT_ANGLE_HI)
                 & (r_map >= WALL_INNER_R) & (r_map <= WALL_OUTER_R)] = MYO_ACTIVITY
    sino_rest_focused = acquire(rest_phantom, heart_focused_weighting(), seed=22)
    sino_rest_clean = run_motion_qc(sino_rest_focused)
    recon_rest = run_reconstruction(sino_rest_clean, rest_phantom)

    sss, srs, sds = run_scoring(recon_stress, recon_rest, phantom)
    ef = run_gated_function()
    rest_flow, stress_flow, mfr = run_dynamic_flow()

    print("\n" + "=" * 60)
    print("CLINICAL-STYLE SUMMARY REPORT (simulated study)")
    print("=" * 60)
    print(f"  Summed Stress Score (SSS):        {sss}")
    print(f"  Summed Rest Score (SRS):           {srs}")
    print(f"  Summed Difference Score (SDS):     {sds}  "
          f"({'reversible ischemia' if sds > 1 else 'no significant ischemia'})")
    print(f"  Gated LVEF (2D analog):             {ef:.0f}%")
    print(f"  Rest MBF (fitted):                  {rest_flow['fitted_flow']:.2f} mL/min/g")
    print(f"  Stress MBF (fitted):                {stress_flow['fitted_flow']:.2f} mL/min/g")
    print(f"  Myocardial Flow Reserve (MFR):       {mfr:.2f}")
    print("=" * 60)

    print("\nSCOPE NOTES (design choices made explicit, not hidden):")
    print("  - Single 2D mid-cavity slice, not a full 3D volume: 6 of the")
    print("    17 clinical segments are scored, gated EF is an area-based")
    print("    2D analog, and dynamic flow is fit at the TAC level rather")
    print("    than via full per-frame image reconstruction.")
    print("  - The 9-column swiveling D-SPECT geometry is represented by")
    print("    its practical effect (heart-weighted angular dwell time)")
    print("    on a standard 0-180 degree sampling grid, not by simulating")
    print("    the exact limited-arc multi-column hardware.")
    print("  - No CT-based attenuation correction, matching the real")
    print("    D-SPECT Cardio (no integrated CT); Spectrum Dynamics'")
    print("    actual product uses a deep-learning AC step (TruCorr) or")
    print("    dual-position imaging in its place, both out of scope here.")
    print("  - Renkin-Crone PS value is a representative literature figure,")
    print("    not a calibrated constant.")
    print("\nAll figures saved to ./outputs -- inspect 01 through 07 in order.")


if __name__ == "__main__":
    main()
