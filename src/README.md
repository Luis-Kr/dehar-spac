src/dehar/
├── __init__.py
│
├── atmosphere/
│   ├── __init__.py
│   ├── flux.py                  # gap-filling, daily aggregation, partitioning QC
│   └── meteo.py                 # VPD calculation, precip accumulation, T filtering
│
├── soil/
│   ├── __init__.py
│   └── swc.py                   # depth averaging, QC, daily aggregation
│
├── physiology/
│   ├── __init__.py
│   ├── water_potential.py       # filtering, species separation, threshold detection
│   ├── sapflow.py               # baseline correction, daily integration
│   └── twd.py                   # normalization, QC
│
├── proximal/
│   ├── __init__.py
│   ├── leaf.py                  # pylidar-tls-canopy wrapper: scan → gap frac → PAVD → PAI
│   ├── anglecam.py              # image → angle timeseries extraction
│   ├── gnss_vod.py              # raw GNSS → transmissivity → VOD
│   └── phenocam.py              # ROI extraction, GCC computation
│
├── satellite/
│   ├── __init__.py
│   ├── sentinel1.py             # backscatter extraction, speckle filtering
│   ├── sentinel2.py             # atmospheric correction check, index computation
│   └── modis.py                 # QA filtering, smoothing, index extraction
│
└── utils/
    ├── __init__.py
    ├── io.py                    # standardized read/write (CSV, NetCDF, LAS)
    ├── time.py                  # temporal alignment, resampling, common time axis
    ├── qc.py                    # generic QC: outlier detection, gap flagging
    └── constants.py             # site coords, species codes, height thresholds