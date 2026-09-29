"""Top‑level package for **attention_matrix_xai_workbench**.

The public API currently consists of:

* :class:`~attention_matrix_xai_workbench.data.CausalSignalGenerator`
* :class:`~attention_matrix_xai_workbench.model.MultiheadPooledAttention`

These are re‑exported here for convenience so users can simply write::

    from attention_matrix_xai_workbench import CausalSignalGenerator, MultiheadPooledAttention

instead of importing the submodules individually.
"""

from .data import CausalSignalDataset, CausalSignalGenerator
from .model import MultiheadPooledAttention

__all__: list[str] = [
    "CausalSignalDataset",
    "CausalSignalGenerator",
    "MultiheadPooledAttention",
]
