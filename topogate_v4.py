import numpy as np
import networkx as nx
from scipy.spatial import KDTree


class TopoGateV4:
    def __init__(self, window=200, stride=20, embed_dim=2, k=5, mode="proxy"):
        self.window = window
        self.stride = stride
        self.embed_dim = embed_dim
        self.k = k
        self.mode = mode
        # Calibration (all learned)
        self.q99_dist = self.q50_circle = self.q10_circle = None
        self.diam_ref = self.signal_var_ref = None
        self.calibrated = False
        self.buffer = None
        if mode == "ph":
            raise NotImplementedError(
                "PH mode not yet implemented – use 'proxy' or swap _extract_features"
            )

    def _embed_scalar(self, ts):
        n = len(ts) - (self.embed_dim - 1)
        return np.column_stack([ts[i : i + n] for i in range(self.embed_dim)])

    def _extract_features(self, data):
        if self.mode != "proxy":
            raise NotImplementedError("PH mode not yet implemented")
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
            eps = diam * 0.5 + 1e-8
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

    def calibrate(self, stable_data):
        """stable_data: 1D or (T, d) — learns everything"""
        raw = np.asarray(stable_data)
        if raw.shape[0] < self.window:
            raise ValueError("Calibration data too short for configured window")
        feats = self._extract_features(raw)  # internal embed only for scalar
        self.signal_var_ref = np.var(raw) + 1e-8  # RAW variance — fixed
        self.diam_ref = np.median(feats[:, 3]) + 1e-8
        circle_raw = feats[:, 1] * feats[:, 3] / self.diam_ref
        circle_str = circle_raw / np.sqrt(self.signal_var_ref)
        self.q99_dist = np.percentile(feats[:, 2], 99)
        self.q50_circle = np.percentile(circle_str, 50)
        self.q10_circle = np.percentile(circle_str, 10)
        self.calibrated = True
        print(
            f"Calibrated → q99_dist={self.q99_dist:.4f} | "
            f"q50_circle={self.q50_circle:.4f} | diam_ref={self.diam_ref:.4f}"
        )
        return self

    def _make_decision(self, feats, signal_var):
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
        """Real-time streaming — Hardstop ready"""
        new_arr = np.asarray(new_batch)
        if new_arr.ndim == 1:
            new_arr = new_arr.reshape(-1, 1)
        if self.buffer is None:
            self.buffer = new_arr
        else:
            self.buffer = np.vstack([self.buffer, new_arr])
        if len(self.buffer) > self.window * 2:
            self.buffer = self.buffer[-(self.window * 2) :]
        if len(self.buffer) < self.window:
            return "WARMUP", 0.0, 0.0
        feats = self._extract_features(self.buffer)
        decisions, circle_str, flare_d = self._make_decision(feats, np.var(self.buffer))
        return decisions[-1], circle_str[-1], flare_d[-1]
