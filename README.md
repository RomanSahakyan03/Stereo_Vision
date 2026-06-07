# Calibration-Free Stereo Depth via a Trainable γ-Grid

A pure-NumPy stereo depth pipeline that replaces classical pinhole intrinsics with a
learnable anisotropic $\gamma$-grid. The system is calibrated from a handful of
user-clicked points whose real-world distances are known — no checkerboard, no OpenCV
`stereoCalibrate`, no automatic differentiation.

Computer Vision course project, 2026.

---

## TL;DR

```
python stereo_app.py left/calib_left.jpg right/calib_right.jpg
```

Click matching points on the two images. Press `K` to add a calibration point at a known
distance. After 15 calibration clicks, the model auto-trains (Adam, ~1 second) and the
saved calibration enables real-time depth estimation on subsequent clicks.

On our 43-point calibration set ($Z = 1.35$–$8.75$ m), leave-one-out mean absolute depth
error drops from $0.373$ m (pinhole baseline) to $0.269$ m (γ-grid + post-correction) —
a $28\%$ improvement. Full numbers in `presentation/report.md`.

---

## Repository layout

```
Stereo_Vision/
├── stereo_app.py                  Full interactive system (training + UI)
├── calibration_points.json        43 hand-clicked (uL, vL, uR, vR, Z_true) rows
├── stereo_calibration.json        Saved γ-grid + baseline + post-correction
├── project_requirements.md        Course assignment spec
├── README.md                      This file
│
├── left/                          Left-camera images (raw 2592×1944)
│   ├── calib_left.jpg               calibration scene
│   ├── scene01_left.jpg … scene03_left.jpg
│   └── test01_left.jpg, test02_left.jpg
├── right/                         Right-camera images (paired by suffix)
│   └── …
│
├── docs/                          Source math derivations (LaTeX)
│   ├── baseline_classical_stereo.tex   classical pinhole + OLS triangulation
│   └── final_gamma_grid_method.tex     full γ-grid method
│
├── experiments/
│   ├── run_experiments.py         LOO cross-validation: pinhole vs γ-grid vs γ-grid+postcorr
│   └── results.json               numerical results (created after running)
│
└── presentation/
    ├── slides_outline.md          15-slide outline + 6 backup slides
    ├── report.md                  2-3 page markdown report
    └── report.tex                 full combined LaTeX report
```

## Requirements

- Python 3.8+
- `numpy`
- `pygame` (interactive UI only — not needed for the experiments script)

```
pip install numpy pygame
```

## Running the interactive app

```
python stereo_app.py <left_image> <right_image>
```

Example:
```
python stereo_app.py left/calib_left.jpg right/calib_right.jpg
```

If `stereo_calibration.json` exists it is loaded at startup and depth estimation is live
immediately. Otherwise the app starts uncalibrated and you must add at least
`MIN_CAL_PTS = 15` clicks before training fires.

### UI controls

| Input | Action |
|---|---|
| Left-click on left image | Pick the left point; draws the epipolar line in the right image and runs a ZNCC sub-pixel match |
| Left-click on right image | Pick / confirm the right point. Snaps to the NCC hint if within 8 px, else orthogonal-projects onto the epipolar line |
| **K** | Calibration mode. Terminal prompts for $Z$ in meters, then click left then right |
| **U** | Undo the last calibration point (or cancel a pending K-mode entry) |
| **C** | Clear all calibration points and reset the model |
| **R** | Clear stored measurements (keeps the trained calibration) |
| **T** | Swap left/right images |
| Mouse wheel | Zoom the panel under the cursor |
| Right-click + drag | Pan the panel |
| **Esc** | Quit |

Training fires automatically every time `len(cal_pts) >= 15` after a calibration click, and
`stereo_calibration.json` is rewritten. The calibration log
(`calibration_points.json`) is rewritten on every add/undo so it survives crashes.

## Reproducing the experiments

```
python experiments/run_experiments.py
```

Runs:
1. **In-sample fit** — trains each of the three configurations on all 43 calibration
   points and reports MAE / MAPE / max error / near (< 3 m) / far (≥ 3 m).
2. **Leave-one-out cross-validation** — 43 folds, ~7 minutes on a typical laptop.

Results are printed as two tables and saved to `experiments/results.json`.

Expected output (depths in meters):

| Configuration | LOO MAE | LOO MAPE | LOO Max | Near MAE | Far MAE |
|---|---|---|---|---|---|
| Pinhole OLS | 0.373 | 8.36 % | 2.56 | 0.180 | 0.448 |
| γ-grid | 0.300 | 9.04 % | 1.79 | 0.373 | 0.271 |
| γ-grid + post-corr | **0.269** | 8.62 % | 1.85 | 0.372 | **0.229** |

## How it works (one paragraph)

Each pixel $(u, v)$ is converted to a normalized 3-D ray $r = [u_n \gamma_x(u,v),\,
v_n \gamma_y(u,v),\, 1]$, where $\gamma_x, \gamma_y$ come from a $3\times3\times2$
trainable grid via bilinear interpolation (separate $x$ and $y$ planes = anisotropic).
Triangulation between the two rays is the classical OLS skew-line solve
$x = (A^\top A)^{-1} A^\top d$. Training minimizes a unit-less **relative disparity loss**
$L_{1D} = \tfrac12(\,D\,Z_\text{true}/d_x - 1)^2$ plus a **Y-epipolar loss** $L_Y$, both
**Huber-clipped** at $\delta = 0.05$ for outlier-click robustness. All gradients are
hand-derived analytically and optimized with Adam plus anchor + Laplacian regularization.
A final quadratic $Z_\text{true} = a + bZ + cZ^2$ absorbs residual systematic bias.

Full derivations are in `docs/baseline_classical_stereo.tex` (the OLS baseline) and
`docs/final_gamma_grid_method.tex` (the γ-grid method). The combined paper version is
`presentation/report.tex`.

## Data format: `calibration_points.json`

JSON array of 5-element lists, one per click:

```json
[
  [uL,   vL,   uR,     vR,     Z_true],
  [1053, 1043, 562,    1036,   1.38  ],
  [2043, 808,  1907,   813,    8.70  ]
]
```

`uL, vL` are integer left-image pixels; `uR, vR` are sub-pixel right-image coordinates
(after ZNCC refinement or orthogonal epipolar snap); `Z_true` is in meters, measured with
a tape. Coordinates are in raw 2592 × 1944 pixels, **not** display panel coordinates.

The app rewrites this file on every K-mode add and every `U` (undo). You may edit it by
hand to remove bad rows.

## Output: `stereo_calibration.json`

Generated by `train()`. Contains:
- `gamma_L_grid`, `gamma_R_grid` — the two $3\times3\times2$ anisotropic grids
- `baseline` — recovered $d = (d_x, d_y, 0)$ in meters
- `postcorr_z` — quadratic coefficients $(a, b, c)$

## Limitations

- **Sparse output.** Depth is computed only for clicked points; the system does not
  produce a dense disparity map. SGBM-style dense reconstruction would require a
  rectification pass that we have not built.
- **Requires ground-truth distances during calibration.** ≥ 15 click pairs with a
  tape-measured $Z$ are needed before the system is useful.
- **Near-region overfitting on small calibration sets.** With 38 parameters and ~43
  points the γ-grid LOO MAE is roughly $2\times$ worse on near points than the pinhole
  baseline. Documented in §5 of the report.
- **One stereo rig validated.** Numbers above are for a single physical camera pair;
  generalization to other rigs requires recollecting calibration data.

## Further reading

- `presentation/report.tex` — combined paper with full math, experiments, and discussion.
- `presentation/report.md` — short markdown version.
- `presentation/slides_outline.md` — talk script.
- `docs/baseline_classical_stereo.tex`, `docs/final_gamma_grid_method.tex` — original
  derivations.

## Author

Roman Sahakyan, 2026.
