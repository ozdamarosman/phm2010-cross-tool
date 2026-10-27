# NASA Ames milling data: external analysis (design fixed before running, 2026-10-27)

Purpose: test transfer to unseen cutting conditions (Reviewer 3, comment 8) with the same leakage-free logic.

Data: mill.mat (Agogino & Goebel 2007), 167 runs, 16 cases = 8 settings (DOC x feed x material) x 2 insert sets.
Units: runs with a VB label. Case 6 (1 run) excluded. Signals at 250 Hz; vib and AE are hardware RMS (8 ms).
Settings: S1 (1.5,0.5,CI)=cases 1,9; S2 (0.75,0.5,CI)=2,12; S3 (0.75,0.25,CI)=3,11; S4 (1.5,0.25,CI)=4,10;
          S5 (1.5,0.5,St)=5,16; S6 (1.5,0.25,St)=15; S7 (0.75,0.25,St)=7,13; S8 (0.75,0.5,St)=8,14.

Segmentation: idle = median smcDC of first 500 samples; active = smcDC > idle + 0.5*(p90(smcDC) - idle);
segment = first..last active sample, trimmed by 10 % at both ends.
Features per channel (6 channels): 7 time-domain stats; Welch PSD (nperseg 1024): log band energies at
f_s = 826/60 = 13.77 Hz (+/-1 Hz), 2 f_s (+/-1.5 Hz), tooth passing f_t = 6 f_s = 82.6 Hz (+/-3 Hz);
log ratio E_fs/E_ft; spectral centroid 0-125 Hz; wavelet packet db4 level 3 relative energies (8).
=> 21 features per channel, 126 in total.

Outer loops: (A, primary) leave-one-setting-out (8 folds; both replicates held out);
             (B, secondary) leave-one-case-out (15 folds).
Inner split: grouped by case within the training pool (GroupKFold, 5 folds), for selection and tuning.
Feature selection: as in PHM2010 (|Pearson| ranking on training pool, forward selection up to 20, Ridge alpha=1 proxy).
Baselines: (i) time-only: isotonic vs cubic on time, chosen on inner split;
           (ii) process baseline: time, DOC, feed, material -> best of {Ridge with time x parameter interactions,
                Random Forest (same grid)} chosen on inner split.
Sensor models: same 7 learners and grids, inputs = selected sensor features + time + DOC + feed + material
               (primary: does sensing add to known process information); sensors-only as secondary.
Metric: RMSE (mm) pooled over held-out runs, per fold. CI: cluster bootstrap over held-out cases (B = 2000), paired.
Criterion: model better than the process baseline if the pooled CI of the difference excludes zero.
Post-hoc normalisation variant: features relative to the first run of each case (n_cal = 1), same pipeline.
No 1D-CNN (few labelled runs, RMS signals). No threshold-crossing analysis (irregular, sparse VB measurements).
Reported whatever the result.

## Data-quality decisions (made after feature extraction, before any modelling)
- smcDC is excluded: it saturates at 10 V in 44 runs (fully clipped in 18), so it is censored. 5 channels -> 105 features.
- Case 12 run 1 is a corrupted record (all channels flat or out of range); it is unlabelled and is not used,
  also not as calibration run (the first valid run of a case is used for the normalisation variant).
- AE_table drops to ~0 in runs 6/1, 7/4 and 8/3 (sensor dropout); kept as recorded.
- Implementation detail: the Ridge proxy for forward selection of sensor features also receives time, DOC,
  feed and material, so that sensor features are selected for what they add to the process information.
