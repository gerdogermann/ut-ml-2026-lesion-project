# Experiments

| # | Date | Issue | Model | Config | Split | Metric(s) | Result | Notes |
|---|------|-------|-------|--------|-------|-----------|--------|-------|
| | | | | | | | | |


# Imre's test stuff (some mock data with some real data features) (2026-10-22; needs formatting):

$ # 1. Ran unit tests (hand-computed values + example lesion check)
PYTHONPATH=src pytest

$ # 2. Ran baselines on the 2 real example .npz files in the repo (Mean error for Baseline A will be 14.04 mm!)
PYTHONPATH=src python scripts/baselines.py data/example_lesions/*.npz

$ # 3. Ran baselines in --mock mode to verify the full Val & Test table formatting
PYTHONPATH=src python scripts/baselines.py --mock

==================================================================================== 

test session starts 

=====================================================================================

platform linux -- Python 3.14.5, pytest-9.1.1, pluggy-1.7.0

rootdir: /gpfs/helios/home/saksimre/ut-ml-2026-lesion-project

configfile: pyproject.toml

collected 2 items


tests/test_metrics.py ..                                                                                                                                                               [100%]

===================================================================================== 

2 passed in 0.44s 

======================================================================================

Train mean target_mm (x, y, z): [ 9.485  4.491 -7.952] mm


| Split | Baseline | Subset | N | Median error (mm) | Mean error (mm) | MEDx (mm) | MEDy (mm) | MEDz (mm) | CPM@5 (%) | CPM@10 (%) | CPM@20 (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| example_lesions | Baseline A (Zero) | Overall | 2 | 14.04 | 14.04 | 9.48 | 4.49 | 9.19 | 50.0% | 50.0% | 50.0% |
| example_lesions | Baseline A (Zero) | has_propagated=True | 2 | 14.04 | 14.04 | 9.48 | 4.49 | 9.19 | 50.0% | 50.0% | 50.0% |
| example_lesions | Baseline B (Train Mean) | Overall | 2 | 13.39 | 13.39 | 9.03 | 3.62 | 9.19 | 0.0% | 0.0% | 100.0% |
| example_lesions | Baseline B (Train Mean) | has_propagated=True | 2 | 13.39 | 13.39 | 9.03 | 3.62 | 9.19 | 0.0% | 0.0% | 100.0% |
Train mean target_mm (x, y, z): [ 0.117 -0.945  0.661] mm

| Split | Baseline | Subset | N | Median error (mm) | Mean error (mm) | MEDx (mm) | MEDy (mm) | MEDz (mm) | CPM@5 (%) | CPM@10 (%) | CPM@20 (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val (mock) | Baseline A (Zero) | Overall | 60 | 6.25 | 7.62 | 3.51 | 4.56 | 3.19 | 33.3% | 88.3% | 95.0% |
| val (mock) | Baseline A (Zero) | has_propagated=True | 50 | 5.85 | 5.84 | 2.84 | 3.24 | 2.54 | 38.0% | 98.0% | 100.0% |
| val (mock) | Baseline A (Zero) | has_propagated=False | 10 | 14.85 | 16.50 | 6.82 | 11.14 | 6.43 | 10.0% | 40.0% | 70.0% |
| val (mock) | Baseline B (Train Mean) | Overall | 60 | 6.04 | 7.45 | 3.49 | 4.34 | 3.20 | 33.3% | 85.0% | 95.0% |
| val (mock) | Baseline B (Train Mean) | has_propagated=True | 50 | 5.62 | 5.72 | 2.83 | 3.06 | 2.61 | 38.0% | 94.0% | 100.0% |
| val (mock) | Baseline B (Train Mean) | has_propagated=False | 10 | 14.60 | 16.10 | 6.80 | 10.76 | 6.17 | 10.0% | 40.0% | 70.0% |
| test (mock) | Baseline A (Zero) | Overall | 60 | 5.65 | 7.60 | 3.88 | 4.16 | 3.44 | 38.3% | 86.7% | 91.7% |
| test (mock) | Baseline A (Zero) | has_propagated=True | 53 | 5.43 | 5.59 | 2.71 | 3.24 | 2.46 | 43.4% | 96.2% | 100.0% |
| test (mock) | Baseline A (Zero) | has_propagated=False | 7 | 24.20 | 22.82 | 12.76 | 11.13 | 10.92 | 0.0% | 14.3% | 28.6% |
| test (mock) | Baseline B (Train Mean) | Overall | 60 | 5.71 | 7.43 | 3.87 | 3.91 | 3.44 | 38.3% | 86.7% | 91.7% |
| test (mock) | Baseline B (Train Mean) | has_propagated=True | 53 | 5.50 | 5.45 | 2.70 | 2.98 | 2.47 | 43.4% | 96.2% | 100.0% |
| test (mock) | Baseline B (Train Mean) | has_propagated=False | 7 | 23.47 | 22.46 | 12.74 | 10.92 | 10.82 | 0.0% | 14.3% | 28.6% |
$ 
