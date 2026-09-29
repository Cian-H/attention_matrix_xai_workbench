# Attention Matrix XAI Workbench

A lightweight Python library for generating synthetic time‑series data and
experimenting with attention‑based models in **marimo** notebooks.

## Table of Contents

- [Installation](#installation)
- [Core Components](#core-components)
  - [`CausalSignalGenerator`](#causalsignalgenerator)
  - [`MultiheadPooledAttention`](#multiheadpooledattention)
- [Running an Experiment](#running-an-experiment)
- [Creating a New Experiment](#creating-a-new-experiment)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Installation

```bash
# Enter the development shell (installs uv, python, and deps)
devenv shell
# First entry creates a virtual‑env and installs the package
uv sync
```

The package is available as an editable install, so imports work out‑of‑the‑box:

```python
from attention_matrix_xai_workbench import CausalSignalGenerator, MultiheadPooledAttention
```

## Core Components

### `CausalSignalGenerator`

*Generates multivariate time‑series data.*

```python
from attention_matrix_xai_workbench import CausalSignalGenerator

schema = {
    "signal_a": {"type": "source", "waveform": "sine", "freq_range": (5, 15)},
    "signal_b": {"type": "mix", "depends_on": {"signal_a": "ramp_up"}},
}

gen = CausalSignalGenerator()
X, y, cols = gen.generate(schema)
```

- **`X`** -- `(n_samples, n_columns)` observation matrix.
- **`y`** -- target vector derived from the STFT of the latent sources.
- **`cols`** -- ordered list of column names.

### `MultiheadPooledAttention`

*A multi‑head cross‑attention module that pools a variable‑length sequence into
a fixed‑size representation.*

```python
import torch
from attention_matrix_xai_workbench import MultiheadPooledAttention

x = torch.randn(batch_size, seq_len, embed_dim)
attn = MultiheadPooledAttention(embed_dim, num_queries=4, attn_activation="softmax")
out = attn(x)  # shape: (batch_size, embed_dim)
```

The module supports custom activation functions (e.g., `"linear"`) and
automatically handles dropout, lagging, and mixing of source signals.

## Running an Experiment

All notebooks belong under the **`notebooks/`** directory. A minimal example is
provided in `notebooks/example.py`.

```bash
uv run marimo edit notebooks/example.py
```
The notebook demonstrates: 1. Generating synthetic data with
`CausalSignalGenerator`. 2. Feeding the data to `MultiheadPooledAttention`. 3.
Visualising attention maps.

## Creating a New Experiment

1. **Create a notebook** in `notebooks/` (e.g., `my_experiment.py`).

2. Import the utilities:
   ```python
   from attention_matrix_xai_workbench import CausalSignalGenerator, MultiheadPooledAttention
   ```

3. Define a **schema** describing the causal DAG of your signals (see the
   example above).

4. Generate the dataset and split it as needed.

5. Instantiate the attention module and train/evaluate your model.

6. Use `marimo` widgets (`mo.ui`, `mo.mpl`, etc.) for interactive visualisation.

## Testing

The repository includes a test suite using **pytest** and **ruff** for linting.

```bash
uv run pytest          # run unit tests
uv run ruff check .    # lint the codebase
uv run ruff format .   # auto‑format
```

Tests are located under `tests/` and cover both data generation and model
forward passes.

## License

Distributed under the LGPLv3 License. See `license.md` for details.
