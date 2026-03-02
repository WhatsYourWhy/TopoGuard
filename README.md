# TopoGuard — Topology-Gated Refusal Layer

TopoGuard is a lightweight, calibration-driven policy operator for dynamical systems. It learns the topological structure of a known stable regime and classifies incoming data into one of three states:

| Decision | Meaning |
|---|---|
| `AUTOMATE` | System is in a stable, recurrent regime (Circle) |
| `INTERVENE/FLARE` | Structural regime change detected (Flare) |
| `REFUSE/VOID` | System is in a non-admissible or uncalibrated state (Void) |

No hard-coded thresholds. No model of the underlying system required. All decisions derive from topological signatures extracted from sliding windows of the input.

---

## Why Topology?

Variance-based detectors (CUSUM, spectral entropy) measure *amplitude*. TopoGuard measures *shape* — specifically, the persistent structure of the system's state-space geometry. A signal can have stable variance while undergoing a fundamental regime change; topology catches what amplitude misses.

The core invariants are:

- **H₀** — connected components (fragmentation)
- **H₁** — persistent cycles (recurrence / limit-cycle structure)
- **Diagram distance** — window-to-window topological change

---

## Highlights

- No hard-coded decision constants — all thresholds are calibrated from your data
- **Proxy backend** (NumPy + NetworkX + SciPy) — lightweight, broadly compatible
- **PH backend** (giotto-tda) — exact persistent homology via Vietoris–Rips filtrations and 1-Wasserstein distances; same interface
- Scalar, multivariate, and streaming support
- Orthogonal to variance-only detectors

---

## Install

```bash
pip install numpy networkx scipy
```

Optional PH backend:

```bash
pip install giotto-tda
```

---

## Quickstart

### Scalar time series (proxy mode)

```python
import numpy as np
from topoguard_v5 import TopoGuardV5

# Calibrate on a known stable regime
stable = np.sin(np.linspace(0, 20, 1000))
model = TopoGuardV5(window=200, stride=20, mode="proxy").calibrate(stable)

# Detect on new data
decision, circle_strength, flare_score = model.detect_scalar(stable[-400:])
print(decision, circle_strength, flare_score)
# → AUTOMATE  0.847  0.002
```

### Multivariate data

```python
X = np.random.randn(1000, 4)  # (timesteps, features)
model = TopoGuardV5(window=100, stride=10, mode="proxy").calibrate(X)
decision, cs, fs = model.detect_vector(X[-200:])
```

### Streaming (online)

```python
model = TopoGuardV5(window=200, stride=20).calibrate(stable)

for batch in incoming_stream:
    decision, cs, fs = model.update(batch)
    if decision != "AUTOMATE":
        alert(decision, cs, fs)
```

---

## Decision Policy

After calibration, each window is classified by three features: **circle strength** (C), **diagram distance** (D), and **H₀ component count**.

```
Circle  →  C ≥ Q50(C_cal)  AND  D ≤ Q99(D_cal)   →  AUTOMATE
Flare   →  D > Q99(D_cal)  OR   H₀ > 2            →  INTERVENE/FLARE
Void    →  otherwise                               →  REFUSE/VOID
```

All quantiles are computed from the calibration regime. Nothing is hard-coded.

---

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `window` | 200 | Sliding window length (samples) |
| `stride` | 20 | Step between windows |
| `embed_dim` | 2 | Takens embedding dimension (scalar mode) |
| `k` | 5 | k-nearest neighbors for proxy graph |
| `mode` | `"proxy"` | `"proxy"` or `"ph"` |

**Minimum data length for calibration:**
- Scalar: `window + (embed_dim - 1)` samples
- Multivariate: `window` samples

---

## Backends

### Proxy mode (default)
Uses a k-nearest-neighbor graph and cycle basis to approximate H₀ and H₁. O(W log W) per window. No additional dependencies beyond NumPy, NetworkX, and SciPy.

### PH mode
Uses Vietoris–Rips persistent homology (via giotto-tda) and 1-Wasserstein diagram distances. More precise, O(W²) per window. Requires `giotto-tda`.

```python
model = TopoGuardV5(window=100, stride=10, mode="ph").calibrate(stable)
```

`mode="ph"` raises a clear `ImportError` if giotto-tda is not installed.

---

## Notes

- A safety floor (`diam = max(diam, 1e-6)`) prevents proxy collapse on near-fixed-point windows
- Scalar calibration and detection consistently use raw input variance for normalization
- Calibration is performed once and reused for all subsequent detections
- The rolling buffer in streaming mode is capped at `2 × window` to bound memory

---

## Roadmap

- [ ] PyPI package (`pip install topoguard`)
- [ ] Persistence diagram visualization
- [ ] Multi-channel / multi-sensor support
- [ ] Benchmark suite vs. CUSUM and spectral entropy

---

## License

MIT
