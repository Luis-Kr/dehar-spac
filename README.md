# DE-Har SPAC Drought Cascade

## Workflow Summary
data/raw/  ──→  scripts/process_*.py  ──→  data/processed/
                (uses src/dehar/)
                                              │
                                              ▼
                                        analysis/
                                              │
                                              ▼
                                        figures/

Raw data lands in data/raw/{domain}/{sensor}/
scripts/process_*.py calls functions from src/dehar/ to produce data/processed/
analysis/ reads from data/processed/ across all streams
figures/ reads from analysis/ outputs and data/processed/