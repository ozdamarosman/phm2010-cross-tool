# PHM2010 cross-tool wear prediction under limited labels

Code for: *Sensor Features Versus the Wear–Time Trend: A Leakage-Free Cross-Tool Evaluation of Tool-Wear Prediction Under Limited Labels on the PHM2010 Milling Benchmark.*

## Steps
1. Download the original 50 kHz PHM2010 data (IEEE DataPort, doi:10.21227/jdxd-yy51) and run
   `python phm2010_extract.py <path-to-archive>`; copy the resulting `phm2010_out/` folder next to these scripts.
2. `pip install numpy pandas scipy scikit-learn xgboost torch shap matplotlib python-docx PyWavelets`
3. Run, in this order:
   ```
   python run_main.py VB_max_um        # nested LOCO, baselines, classical learners, bootstrap
   python run_main.py VB_mean_um       # sensitivity analysis (Table S2)
   python run_cnn.py VB_max_um 20      # 1D-CNN
   python cnn_summary.py max
   python run_extras.py crossing VB_max_um
   python run_extras.py explain VB_max_um
   python run_extras.py ablation VB_max_um
   python run_extras.py lcurve VB_max_um
   python make_figs.py
   python make_supp.py
   ```
Results are written to `results/`, figures to `figs/`. Seeds 0–4 are fixed in the code.

## Post-hoc analysis (per-tool normalisation, Section 3.7 / 4.7)
Design fixed before running: see `PREREG_normalisation.md`.
```
python run_norm.py main 5
python run_norm.py main 1
python run_norm.py main 10
python run_norm.py extras 5
```

## External data: NASA Ames milling (Section 3.8 / 4.8)
Download mill.zip from https://data.nasa.gov/dataset/milling-wear and put mill.mat in `nasa/`. Design: `nasa/PREREG_nasa.md`.
```
cd nasa
python nasa_features.py mill.mat nasa_features.csv
python nasa_model.py setting raw
python nasa_model.py setting norm
python nasa_model.py case raw
python nasa_fig.py
```
