import importlib.util

import networkx as nx
import numpy as np
from scipy.spatial import KDTree


class TopoGuardV5:
    """
    Topological refusal layer.

    - mode="proxy": kNN-graph + cycle basis (lightweight)
    - mode="ph": Vietoris–Rips + Wasserstein via giotto-tda

    Public API:
      - calibrate(stable_data): learn topological envelope
      - detect_scalar(ts_1d): batch decision on 1D series
      - detect_vector(X): batch decision on (T, d) multivariate data
      - update(new_batch): streaming decision with rolling buffer
    """

    def __init__(self, window=200, stride=20, embed_dim=2, k=5, mode="proxy"):
        self.window = window
        self.stride = stride
        self.embed_dim = embed_dim
        self.k = k
        self.mode = mode

        self.q99_dist = self.q50_circle = self.q10_circle = None
        self.diam_ref = self.signal_var_ref = None
        self.calibrated = False
        self.buffer = None

        if mode not in {"proxy", "ph"}:
            raise ValueError("mode must be 'proxy' or 'ph'")
        if mode == "ph" and importlib.util.find_spec("gtda") is None:
            raise ImportError("giotto-tda not installed. Run: pip install giotto-tda")

    def _embed_scalar(self, ts):
        n = len(ts) - (self.embed_dim - 1)
        if n <= 0:
            raise ValueError("Input too short for embedding dimension")
        return np.column_stack([ts[i : i + n] for i in range(self.embed_dim)])

    def _extract_features_proxy(self, data):
        data = np.asarray(data)
        if data.ndim == 1:
            data = self._embed_scalar(data)

        t_steps = len(data)
        windows = [
            data[i : i + self.window]
            for i in range(0, t_steps - self.window + 1, self.stride)
        ]

        features = []
        prev_h0 = prev_h1 = 0
        for w in windows:
            if len(w) < 10:
                continue

            tree = KDTree(w)
            dists, idx = tree.query(w, k=self.k)
            diam = np.max(dists[:, -1]) if len(dists) > 0 else 0.0
            diam = max(diam, 1e-6)
            eps = diam * 0.5

            graph = nx.Graph()
            graph.add_nodes_from(range(len(w)))
            for i in range(len(w)):
                for j_idx in range(1, self.k):
                    j = idx[i, j_idx]
                    if dists[i, j_idx] < eps:
                        graph.add_edge(i, j)

            h0 = nx.number_connected_components(graph)
            cycles = nx.cycle_basis(graph)
            h1_p = max((len(c) for c in cycles), default=0) / len(w) if cycles else 0.0
            diag_dist = abs(h0 - prev_h0) + abs(h1_p - prev_h1)
            features.append([h0, h1_p, diag_dist, diam])
            prev_h0, prev_h1 = h0, h1_p

        feats = np.array(features) if features else np.zeros((0, 4))
        if feats.shape[0] == 0:
            raise ValueError("No features extracted – increase data length or adjust window/stride")
        return feats

    def _extract_features_ph(self, data):
        from gtda.diagrams import PairwiseDistance
        from gtda.homology import VietorisRipsPersistence
        from gtda.time_series import SlidingWindow

        data = np.asarray(data)
        if data.ndim == 1:
            data = self._embed_scalar(data)

        sw = SlidingWindow(size=self.window, stride=self.stride)
        windows = sw.fit_transform(data)

        persistence = VietorisRipsPersistence(homology_dimensions=[0, 1], n_jobs=-1)
        diagrams = persistence.fit_transform(windows)

        dist = PairwiseDistance(metric="wasserstein")
        diagram_dists = dist.fit_transform(diagrams)

        feats = []
        for i, d in enumerate(diagrams):
            h0_count = np.sum(d[:, 2] == 0) if d.shape[0] else 0

            h1_mask = d[:, 2] == 1 if d.shape[0] else np.array([], dtype=bool)
            if np.any(h1_mask):
                h1_persist = np.max(d[h1_mask, 1] - d[h1_mask, 0])
            else:
                h1_persist = 0.0

            dd = diagram_dists[i - 1, i] if i > 0 else 0.0
            feats.append([h0_count, h1_persist, dd, 1.0])

        feats = np.array(feats) if feats else np.zeros((0, 4))
        if feats.shape[0] == 0:
            raise ValueError("No PH features extracted – increase data length or adjust window/stride")
        return feats

    def _extract_features(self, data):
        if self.mode == "proxy":
            return self._extract_features_proxy(data)
        return self._extract_features_ph(data)

    def _min_samples_needed(self, arr):
        arr = np.asarray(arr)
        if self.mode == "proxy" and arr.ndim == 1:
            return self.window + (self.embed_dim - 1)
        return self.window

    def calibrate(self, stable_data):
        raw = np.asarray(stable_data)
        needed = self._min_samples_needed(raw)
        if raw.shape[0] < needed:
            raise ValueError(
                f"Calibration data too short for configured window; need at least {needed} samples"
            )

        feats = self._extract_features(raw)
        self.signal_var_ref = np.var(raw) + 1e-8
        self.diam_ref = np.median(feats[:, 3]) + 1e-8

        circle_raw = feats[:, 1] * feats[:, 3] / self.diam_ref
        circle_str = circle_raw / np.sqrt(self.signal_var_ref)

        self.q99_dist = np.percentile(feats[:, 2], 99)
        self.q50_circle = np.percentile(circle_str, 50)
        self.q10_circle = np.percentile(circle_str, 10)
        self.calibrated = True

        print(
            f"Calibrated → q99_dist={self.q99_dist:.4f} | "
            f"q50_circle={self.q50_circle:.4f} | diam_ref={self.diam_ref:.4f} | mode={self.mode}"
        )
        return self

    def _make_decision(self, feats, signal_var):
        if not self.calibrated:
            raise RuntimeError("Model must be calibrated before detection")

        circle_raw = feats[:, 1] * feats[:, 3] / self.diam_ref
        circle_str = circle_raw / np.sqrt(signal_var + 1e-8)

        decisions = []
        for i in range(len(feats)):
            cs, dd, h0 = circle_str[i], feats[i, 2], feats[i, 0]
            if cs > self.q50_circle and dd < self.q99_dist:
                decisions.append("AUTOMATE")
            elif dd > self.q99_dist or h0 > 2:
                decisions.append("INTERVENE/FLARE")
            else:
                decisions.append("REFUSE/VOID")

        return decisions, circle_str, feats[:, 2]

    def detect_scalar(self, ts_1d):
        feats = self._extract_features(ts_1d)
        decisions, circle_str, flare_d = self._make_decision(feats, np.var(ts_1d))
        return (
            decisions[-1] if decisions else "UNKNOWN",
            circle_str[-1] if len(circle_str) else 0.0,
            flare_d[-1] if len(flare_d) else 0.0,
        )

    def detect_vector(self, x_data):
        feats = self._extract_features(x_data)
        decisions, circle_str, flare_d = self._make_decision(feats, np.var(x_data))
        return (
            decisions[-1] if decisions else "UNKNOWN",
            circle_str[-1] if len(circle_str) else 0.0,
            flare_d[-1] if len(flare_d) else 0.0,
        )

    def update(self, new_batch):
        new_arr = np.asarray(new_batch)
        if new_arr.ndim == 0:
            new_arr = new_arr.reshape(1)
        elif new_arr.ndim > 2:
            raise ValueError("new_batch must be scalar, 1D, or 2D")

        if self.buffer is None:
            self.buffer = new_arr.copy()
        else:
            if self.buffer.ndim != new_arr.ndim:
                raise ValueError(
                    "new_batch dimensionality does not match existing stream buffer"
                )

            if self.buffer.ndim == 1:
                self.buffer = np.concatenate([self.buffer, new_arr])
            else:
                if self.buffer.shape[1] != new_arr.shape[1]:
                    raise ValueError(
                        "new_batch feature width does not match existing stream buffer"
                    )
                self.buffer = np.vstack([self.buffer, new_arr])

        if len(self.buffer) > self.window * 2:
            self.buffer = self.buffer[-(self.window * 2) :]
        if len(self.buffer) < self._min_samples_needed(self.buffer):
            return "WARMUP", 0.0, 0.0

        feats = self._extract_features(self.buffer)
        decisions, circle_str, flare_d = self._make_decision(feats, np.var(self.buffer))
        return decisions[-1], circle_str[-1], flare_d[-1]
