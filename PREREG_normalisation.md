# Post-hoc analysis: per-tool baseline normalisation (fixed before running, 2026-10-26)

Motivation (from the main analysis): tool-to-tool differences in sensor levels limit cross-tool transfer.

Rule (fixed in advance, no tuning on the held-out cutter):
- Calibration cuts: the first n_cal = 5 cuts of every cutter (training and held-out) are recorded WITHOUT labels.
- Reference r_f = mean of feature f over the calibration cuts of that cutter.
- Normalised feature: if f is strictly positive in the whole data set -> log(x / r_f); otherwise -> x - r_f
  (already log10-transformed energies are not strictly positive and therefore get x - r_f, i.e. a log ratio).
- Target unchanged (VB_max, mm). Everything else identical to the main analysis
  (k = 20, nested selection + tuning, grouped inner split, same grids, seeds 0-4, block bootstrap L = 20, B = 2000).
- Primary comparison: each learner on normalised features vs. the cut-index baseline (same criterion as Section 3.6).
- Secondary: normalised vs. raw features for the same learner; n_cal in {1, 10} as sensitivity;
  learning curve and threshold crossing (VB_lim = 0.15 mm) for the Random Forest.
- The 1D-CNN is not re-run (normalisation is defined on features).
- Reported as a post-hoc analysis, whatever the result.
