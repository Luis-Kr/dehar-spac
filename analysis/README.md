analysis/
├── cascade/
│   ├── changepoint_detection.py     # PELT on each variable
│   ├── lag_structure.py             # onset lags between sensor pairs
│   ├── cross_correlation.py         # windowed cross-correlations
│   └── dampening.py                 # amplitude ratios across cascade stages
│
├── layer_separation/
│   ├── overstory_vs_understory.py   # PAVD-based layer dynamics
│   └── compensating_signals.py      # how layers mask stress in bulk indices
│
└── satellite_comparison/
    ├── ground_vs_space.py           # what satellites see vs. ground truth
    └── index_sensitivity.py         # which index detects what, with what delay