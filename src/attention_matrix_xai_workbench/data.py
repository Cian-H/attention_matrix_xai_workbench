"""Data generation utilities.

This module provides a *CausalSignalGenerator* that can create synthetic
multivariate time‑series data together with a target vector derived from the
latent (noise‑free) sources.  The generator is deliberately configurable – the
caller supplies a *schema* that describes how each column is built (source,
noise, mix, etc.).  The implementation mirrors the original one but now includes
comprehensive docstrings and type annotations so static analysis tools can
understand the API.

Typical usage::

    from attention_matrix_xai_workbench.data import CausalSignalGenerator
    schema = {
        "signal_a": {"type": "source", "waveform": "sine", "freq_range": (5, 15)},
        "signal_b": {"type": "mix", "depends_on": {"signal_a": "ramp_up"}},
    }
    gen = CausalSignalGenerator()
    X, y, cols = gen.generate(schema)

* ``X`` – ``np.ndarray`` of shape ``(n_samples, n_columns)``
* ``y`` – target vector derived from the STFT of the latent sources
* ``cols`` – list of column names in the order used for ``X``
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.signal as signal
import torch
from torch.utils.data import IterableDataset


class CausalSignalGenerator:
    """Generate synthetic causal time‑series data.

    Parameters
    ----------
    n_samples: int, default ``1000``
        Number of time steps in each generated series.
    fs: float, default ``100.0``
        Sampling frequency (Hz).  Used for STFT target extraction.
    global_noise: float, default ``0.05``
        Standard deviation of additive Gaussian noise applied to the final
        observation matrix ``X``.
    """

    def __init__(
        self, n_samples: int = 1000, fs: float = 100.0, global_noise: float = 0.05
    ) -> None:
        self.n_samples: int = n_samples
        self.fs: float = fs
        self.global_noise: float = global_noise
        self.t: np.ndarray = np.linspace(0, n_samples / fs, n_samples)
        self._table: Dict[str, np.ndarray] = {}
        self._latent_sources: List[np.ndarray] = []
        self._resolving_path: set[str] = set()

    def _make_source_signal(self, config: Dict[str, Any]) -> np.ndarray:
        """Create a base waveform according to *config*.

        The function supports three waveforms:
        ``"sine"``, ``"chirp"`` and ``"square"``.  An envelope of intermittent
        Hann‑window bursts is applied to make the signal *causal* (i.e. sparse in
        time).  The generated signal is appended to ``self._latent_sources`` so it
        contributes to the STFT‑derived target.
        """
        freq_range: Tuple[float, float] = config.get("freq_range", (10, 45))
        waveform: str = config.get("waveform", "sine")

        envelope = np.zeros_like(self.t)
        for _ in range(np.random.randint(2, 5)):
            dur = np.random.randint(self.n_samples // 10, self.n_samples // 5)
            start = np.random.randint(0, self.n_samples - dur)
            envelope[start : start + dur] = np.maximum(
                envelope[start : start + dur], signal.windows.hann(dur)
            )

        if waveform == "sine":
            freq = np.random.uniform(*freq_range)
            wave = np.sin(2 * np.pi * freq * self.t)
        elif waveform == "chirp":
            wave = signal.chirp(self.t, f0=freq_range[0], t1=self.t[-1], f1=freq_range[1])
        elif waveform == "square":
            freq = np.random.uniform(*freq_range)
            wave = signal.square(2 * np.pi * freq * self.t)
        else:
            raise ValueError(f"Unknown waveform: {waveform}")

        burst_signal = wave * envelope
        self._latent_sources.append(burst_signal)
        return burst_signal

    def _get_dynamic_weight(self, weight_def: Any) -> np.ndarray | float:
        """Resolve ``weight_def`` to a numeric multiplier.

        ``weight_def`` may be:
        * a plain ``float`` / ``int`` – returned unchanged
        * ``"ramp_up"`` – linearly increases from 0 to 1 over the series
        * ``"fade_out"`` – linearly decreases from 1 to 0
        """
        if isinstance(weight_def, (int, float)):
            return float(weight_def)
        if weight_def == "ramp_up":
            return np.linspace(0, 1.0, self.n_samples)
        if weight_def == "fade_out":
            return np.linspace(1.0, 0, self.n_samples)
        raise ValueError(f"Unknown weight definition: {weight_def}")

    def _apply_dropout(self, sig: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """Simulate sensor failure by zero‑ing random chunks.

        Parameters
        ----------
        sig: np.ndarray
            Signal to modify.
        config: dict
            Expected keys ``"dropout_chunks"`` and ``"dropout_length"``.
        """
        chunks: int = config.get("dropout_chunks", 0)
        length: int = config.get("dropout_length", 50)
        if chunks > 0:
            sig = sig.copy()
            for _ in range(chunks):
                start = np.random.randint(0, max(1, self.n_samples - length))
                sig[start : start + length] = 0.0
        return sig

    def _resolve_column(self, col_name: str, schema: Dict[str, Dict[str, Any]]) -> np.ndarray:
        """Recursively resolve a column based on the *schema* DAG.

        The result is cached in ``self._table`` to avoid recomputation and to
        detect cycles.
        """
        if col_name in self._table:
            return self._table[col_name]
        if col_name in self._resolving_path:
            raise ValueError(f"Cycle detected! Path: {self._resolving_path} -> {col_name}")

        self._resolving_path.add(col_name)
        config = schema[col_name]

        if config["type"] == "source":
            sig = self._make_source_signal(config)
        elif config["type"] == "noise":
            sig = np.zeros(self.n_samples)
        elif config["type"] == "mix":
            interaction: str = config.get("interaction", "add")
            parent_signals: List[np.ndarray] = []
            total_static_weight: float = 0.0
            for parent_col, weight_def in config.get("depends_on", {}).items():
                p_sig = self._resolve_column(parent_col, schema)
                lag: int = config.get("lags", {}).get(parent_col, 0)
                if lag != 0:
                    p_sig = np.roll(p_sig, lag)
                w = self._get_dynamic_weight(weight_def)
                parent_signals.append(p_sig * w)
                if isinstance(w, float):
                    total_static_weight += w
            if interaction == "add":
                sig = sum(parent_signals) if parent_signals else np.zeros(self.n_samples)
                if total_static_weight < 1.0:
                    private_weight = 1.0 - total_static_weight
                    p_config = {
                        "waveform": config.get("waveform", "sine"),
                        "freq_range": config.get("private_freq", (10, 45)),
                    }
                    private_sig = self._make_source_signal(p_config)
                    sig += private_weight * private_sig
            elif interaction == "multiply":
                sig = np.ones(self.n_samples)
                for p in parent_signals:
                    sig *= p
            else:
                raise ValueError(f"Unknown interaction: {interaction}")
        else:
            raise ValueError(f"Unsupported column type: {config['type']}")

        sig = self._apply_dropout(sig, config)

        self._table[col_name] = sig
        self._resolving_path.remove(col_name)
        return sig

    def generate(
        self, schema: Dict[str, Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Generate a full dataset from *schema*.

        Returns
        -------
        X: np.ndarray, shape ``(n_samples, n_columns)``
            Observation matrix.
        target_vector: np.ndarray
            Max magnitude of the STFT of the summed latent sources – this is the
            supervised target used in downstream attention‑matrix experiments.
        columns: list[str]
            Ordered list of column names corresponding to the columns in ``X``.
        """
        self._table = {}
        self._latent_sources = []
        self._resolving_path = set()

        for col_name in schema:
            self._resolve_column(col_name, schema)

        columns = list(schema.keys())
        X = np.column_stack([self._table[col] for col in columns])
        X += np.random.normal(0, self.global_noise, X.shape)

        pure_signal_sum = np.sum(self._latent_sources, axis=0)
        _, _, Zxx = signal.stft(pure_signal_sum, fs=self.fs, nperseg=100)
        target_vector = np.max(np.abs(Zxx), axis=1)

        return X, target_vector, columns


class CausalSignalDataset(IterableDataset):
    """PyTorch IterableDataset for streaming causal time-series data.

    This dataset wraps the ``CausalSignalGenerator`` to synthesize data on-the-fly,
    ensuring the model never sees the exact same training example twice. It automatically
    handles casting numpy arrays to ``torch.float32`` tensors.

    Parameters
    ----------
    generator: CausalSignalGenerator
        An instantiated generator object configured with desired sampling rates and noise.
    schema: dict
        The DAG schema defining the relationships, waveforms, and dropouts.
    samples_per_epoch: int, default ``1000``
        The number of unique examples to generate per epoch.
    """

    def __init__(
        self,
        generator: CausalSignalGenerator,
        schema: Dict[str, Dict[str, Any]],
        samples_per_epoch: int = 1000,
    ) -> None:
        super().__init__()
        self.generator: CausalSignalGenerator = generator
        self.schema: Dict[str, Dict[str, Any]] = schema
        self.samples_per_epoch: int = samples_per_epoch

    def __iter__(self):
        """Yield a single ``(X, y)`` tensor pair per iteration."""
        for _ in range(self.samples_per_epoch):
            X_np, y_np, _ = self.generator.generate(self.schema)
            X_tensor = torch.from_numpy(X_np).float()
            y_tensor = torch.from_numpy(y_np).float()
            yield X_tensor, y_tensor
