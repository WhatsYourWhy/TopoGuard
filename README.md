# TopoGuard — Topology-Gated Refusal Layer

A lightweight, calibration-only policy operator that learns topological invariants from a stable regime and returns:

- `AUTOMATE`
- `INTERVENE/FLARE`
- `REFUSE/VOID`

## Highlights

- No hard-coded decision constants.
- Proxy backend (NumPy + NetworkX + SciPy) for broad compatibility.
- Optional PH backend (`giotto-tda`) with the same calibration/detection interface.
- Scalar + multivariate + streaming support.
- Orthogonal to variance-only detectors such as CUSUM and spectral entropy.

## Install

```bash
pip install numpy networkx scipy
# optional PH backend
pip install giotto-tda
```

## Usage

```python
import numpy as np
from topoguard_v5 import TopoGuardV5

stable = np.sin(np.linspace(0, 20, 1000))
model = TopoGuardV5(window=200, stride=20, mode="proxy").calibrate(stable)

decision, circle_strength, flare_score = model.detect_scalar(stable[-400:])
print(decision, circle_strength, flare_score)
```

## Notes

- A safety floor (`diam = max(diam, 1e-6)`) prevents proxy collapse on near fixed-point windows.
- Scalar calibration/detection consistently use raw input variance.
- `mode="ph"` raises a clear `ImportError` unless `giotto-tda` is installed.
