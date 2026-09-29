import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from attention_matrix_xai_workbench.data import CausalSignalGenerator


@pytest.fixture
def default_generator():
    """Provides a standard generator instance for functional tests."""
    return CausalSignalGenerator(n_samples=1000, fs=100.0, global_noise=0.01)


@settings(deadline=None)
@given(
    n_samples=st.integers(min_value=200, max_value=2000),
    fs=st.floats(min_value=50.0, max_value=500.0),
    global_noise=st.floats(min_value=0.0, max_value=0.5),
)
def test_generator_stability_under_fuzzing(n_samples, fs, global_noise):
    """Fuzzes core parameters to ensure mathematical stability across edge cases."""
    generator = CausalSignalGenerator(n_samples=n_samples, fs=fs, global_noise=global_noise)

    schema = {
        "Src": {"type": "source", "waveform": "sine"},
        "Mix": {"type": "mix", "depends_on": {"Src": 0.5}, "private_freq": (10, 20)},
    }

    X, target, cols = generator.generate(schema)

    # Assert structural properties
    assert X.shape == (n_samples, 2), "Output array shape does not match n_samples"
    assert len(target) > 0, "STFT target vector is empty"
    assert np.isfinite(X).all(), "Generated data contains NaNs or Infs"
    assert np.isfinite(target).all(), "Target vector contains NaNs or Infs"


def test_morphology_and_base_nodes(default_generator):
    """Verifies that all base waveforms and pure noise nodes compile correctly."""
    schema = {
        "Sine_Source": {"type": "source", "waveform": "sine", "freq_range": (10, 15)},
        "Chirp_Source": {"type": "source", "waveform": "chirp", "freq_range": (20, 50)},
        "Square_Source": {"type": "source", "waveform": "square", "freq_range": (5, 8)},
        "Pure_Noise": {"type": "noise"},
    }

    X, _, cols = default_generator.generate(schema)

    assert X.shape == (1000, 4)
    assert "Pure_Noise" in cols

    # The noise column should only contain the global noise
    noise_idx = cols.index("Pure_Noise")
    assert np.isclose(np.std(X[:, noise_idx]), 0.01, atol=0.005)


def test_deep_causal_chains_and_lags(default_generator):
    """Verifies recursive DAG resolution and temporal lags work."""
    schema = {
        "Root": {"type": "source"},
        "Mid_Lagged": {"type": "mix", "depends_on": {"Root": 1.0}, "lags": {"Root": 50}},
        "Leaf_Partial": {
            "type": "mix",
            "depends_on": {"Mid_Lagged": 0.5},
            "private_freq": (40, 45),
        },
    }

    X, target, _ = default_generator.generate(schema)
    assert X.shape == (1000, 3)
    assert len(target) > 0


def test_dynamic_weights_and_nonlinear(default_generator):
    """Verifies time-varying envelopes and multiplicative interactions."""
    schema = {
        "Src_A": {"type": "source", "freq_range": (5, 10)},
        "Src_B": {"type": "source", "freq_range": (20, 25)},
        "Fading_Mix": {"type": "mix", "depends_on": {"Src_A": "ramp_up", "Src_B": "fade_out"}},
        "Intermodulated": {
            "type": "mix",
            "interaction": "multiply",
            "depends_on": {"Src_A": 1.0, "Src_B": 1.0},
        },
    }

    X, _, _ = default_generator.generate(schema)
    assert X.shape == (1000, 4)


def test_sensor_dropout():
    """Verifies the masking logic successfully suppresses signal variance."""
    test_gen = CausalSignalGenerator(n_samples=1000, fs=100.0, global_noise=0.0)

    schema = {
        "Stable_Source": {"type": "source"},
        "Failing_Sensor": {
            "type": "mix",
            "depends_on": {"Stable_Source": 1.0},
            "dropout_chunks": 10,
            "dropout_length": 80,
        },
    }

    X, _, cols = test_gen.generate(schema)

    stable_sig = X[:, cols.index("Stable_Source")]
    failing_sig = X[:, cols.index("Failing_Sensor")]

    var_stable = np.var(stable_sig)
    var_failing = np.var(failing_sig)

    assert var_failing < var_stable, "Dropout failed to reduce variance"

    zeros_stable = np.count_nonzero(stable_sig == 0)
    zeros_failing = np.count_nonzero(failing_sig == 0)
    assert zeros_failing > zeros_stable, "Dropout did not introduce exact zeros"


def test_cycle_detection(default_generator):
    """Ensures the DAG resolver catches circular dependencies and prevents infinite loops."""
    schema_cycle = {
        "Node_A": {"type": "mix", "depends_on": {"Node_B": 1.0}},
        "Node_B": {"type": "mix", "depends_on": {"Node_A": 1.0}},
    }

    with pytest.raises(ValueError, match="Cycle detected"):
        default_generator.generate(schema_cycle)
