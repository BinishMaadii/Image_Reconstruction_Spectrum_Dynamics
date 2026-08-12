# Cardiac SPECT Simulation Pipeline (D-SPECT Model)

A compact Python pipeline simulating cardiac SPECT imaging, inspired by solid-state CZT cameras (e.g., Spectrum Dynamics D-SPECT). Covers the full workflow from phantom acquisition to iterative reconstruction and clinical kinetic modeling.

---

## Key Features

- **Cardiac Phantom & Acquisition**: 2D mid-LV short-axis phantom with an inferolateral defect; compares uniform vs. heart-focused adaptive dwell acquisition under Poisson noise.
- **Motion QC**: Automated Center-of-Mass (COM) tracking with $3\sigma$ residual thresholding and sub-pixel sinogram correction.
- **OSEM Reconstruction**: Iterative reconstruction (8 iters, 6 subsets) with PSF/collimator blur modeling for resolution recovery.
- **Perfusion Scoring**: 6-segment polar analysis computing Summed Stress (SSS), Rest (SRS), and Difference (SDS) scores.
- **Kinetic Flow Modeling (MBF/MFR)**: 1-tissue compartment model with Renkin-Crone flow inversion to estimate Rest/Stress MBF and Flow Reserve.

---

## Installation & Setup

Requires **Python 3.9+**.

```bash
git clone [https://github.com/](https://github.com/)<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install -r requirements.txt

### Create a requirements.txt and write inside it

numpy>=1.22.0
scipy>=1.9.0
scikit-image>=0.19.0
matplotlib>=3.5.0


## Usage
Run the end-to-end pipeline:


