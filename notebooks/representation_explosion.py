import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def overview(mo):
    mo.md(r"""
    This experiment aims to try and use contribution sorted displacement fields to "explode" the entangled representations of the concepts in the attention matrices out into disentangled representations in a higher dimensional space. The basic algorithm would be summarised as follows:

    1. Take the attention matrix (`query_token` in this model) - example shape: (8, 12)

    2. Get an identically shaped array for each output class, scoring the contribution of the numbers in the attention matrices to each class. Then these with the original attention matrix to give a "contribution matrix". e.g: for 4 class classification this would give us a contribution matrix with a shape of (8, 12, 5)

    3. Along dimension 2, we sort each (8, 12) slice and calculate an (8, 12) displacement field along each axis and stack them. This results in an (8, 12, 2) displacement matrix that separates the axes. Then we re-stack those along axis 2 resulting in an (8, 12, 5, 2) matrix. I guess i would call this an "attention contribution displacement matrix"? The visualisation of it would look like an explosion, where every contribution maps onto a unique point in the displacement space though, so i'm just going to refer to it as the "exploded representation" for now.

    This exploded representation should provide a unique fingerprint for how each value in the original attention matrix contributes to the learned representations for each concept/class. The only question then becomes: how do we take that high dimensional, exploded view and collapse it back down to something human interpretable?
    """)
    return


@app.cell
def import_dependencies():
    import marimo as mo
    import matplotlib.pyplot as plt
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    from attention_matrix_xai_workbench.data import CausalSignalDataset, CausalSignalGenerator
    from attention_matrix_xai_workbench.model import MultiheadPooledAttention

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return (
        CausalSignalDataset,
        CausalSignalGenerator,
        DataLoader,
        MultiheadPooledAttention,
        device,
        mo,
        nn,
        plt,
        torch,
    )


@app.cell
def create_schema(mo):
    mo.md("""
    The dataset created to test this proposed technique has 5 features:

    - A: A sine wave
    - B: A square wave with a non-overlapping frequency range
    - C: A signal with ramping up dependent on A
    - D: A signal with ramping up dependent on B
    - E: A signal with ramping up dependency on A and ramping up dependency on B (additive)
    """)

    schema = {
        "A": {"type": "source", "waveform": "sine", "freq_range": (5, 15)},
        "B": {"type": "source", "waveform": "square", "freq_range": (20, 25)},
        "C": {"type": "mix", "depends_on": {"A": "ramp_up"}, "interaction": "add"},
        "D": {"type": "mix", "depends_on": {"B": "ramp_up"}, "interaction": "add"},
        "E": {
            "type": "mix",
            "depends_on": {"A": "ramp_up", "B": "ramp_up"},
            "interaction": "add",
        },
    }
    return (schema,)


@app.cell
def generate_data(
    CausalSignalDataset,
    CausalSignalGenerator,
    DataLoader,
    schema,
):
    generator = CausalSignalGenerator(n_samples=1000)
    dataset = CausalSignalDataset(generator, schema, samples_per_epoch=1000)
    loader = DataLoader(dataset, batch_size=64)
    return dataset, loader


@app.cell
def configure_model(MultiheadPooledAttention, device, nn, torch):
    # Model, loss, optimizer
    model = MultiheadPooledAttention(embed_dim=5).float().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    return criterion, model, optimizer


@app.cell
def train(criterion, dataset, device, loader, mo, model, optimizer, plt):
    losses: list[float] = []
    fig, ax = plt.subplots(figsize=(6, 4))

    for epoch in mo.status.progress_bar(range(50), title="Training Epochs"):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            preds_scalar = preds.mean(dim=1, keepdim=True)
            loss = criterion(preds_scalar, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= dataset.samples_per_epoch
        losses.append(epoch_loss)

        # Update loss curve at end of every epoch
        ax.clear()
        ax.plot(losses, marker="o", color="blue", label="Training Loss")
        ax.set_title(f"Epoch {epoch + 1} | Loss: {epoch_loss:.4f}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True)
        mo.output.replace(fig)
    trained_model = model
    return (trained_model,)


@app.cell
def exploded_representation(device, loader, plt, torch, trained_model):
    from attention_matrix_xai_workbench.attribution import attention_x_gradient

    # Grab a single batch and slice out one observation using private variables
    _xb, _yb = next(iter(loader))
    _xb = _xb.to(device)
    x_single = _xb[0:1]

    # embed_dim represents our 5 features/classes
    embed_dim = 5

    # Generate contribution matrices for each output class.
    attn_contribution_matrices = []
    qt_contribution_matrices = []

    for class_idx in range(embed_dim):
        contrib = attention_x_gradient(
            trained_model,
            x_single,
            target_out_idx=class_idx,
            attention_layer_name="attn_matrix_hookpoint",
        )
        # attn_weights: [1, H, T, S] → mean over heads -> [T, S]
        attn_contribution_matrices.append(contrib.attn_weights[0].mean(dim=0))
        # query_token: [1, T, D] → squeeze batch dim -> [T, D]
        qt_contribution_matrices.append(contrib.query_token[0])

    def _displacement_field(C_matrix):
        """Return an (H, W, C, 2) displacement matrix for a (H, W, C) contribution matrix."""
        H, W, C = C_matrix.shape
        i_grid, j_grid = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        disp = torch.zeros((H, W, C, 2))
        for c in range(C):
            slice_c = C_matrix[..., c]
            # Rank elements by their contribution value (0 = lowest, H*W-1 = highest).
            ranks = slice_c.flatten().argsort().argsort()
            # Map the flat 1D rank of each element to a target (row, col) in the 2D grid.
            # Reshape to (H, W) so the subtraction against i_grid/j_grid is element-wise.
            sorted_i = (ranks // W).reshape(H, W)
            sorted_j = (ranks % W).reshape(H, W)
            # Displacement Field: (Target Position - Original Position)
            disp[..., c, 0] = sorted_i - i_grid
            disp[..., c, 1] = sorted_j - j_grid
        return disp, i_grid, j_grid

    def _plot_attn_row(ax_row, C_matrix, row_label):
        """Show attention-weight contributions as heatmaps.

        The attn-weight grid is [num_queries, seq_len] — e.g. [4, 1000].  That
        extreme aspect ratio makes quiver arrows collapse into horizontal lines
        regardless of pooling, so we use imshow instead.  This directly answers
        "which timesteps does each query attend to most for this concept?".
        """
        _H, _W, C = C_matrix.shape
        for c in range(C):
            ax = ax_row[c]
            ax.imshow(
                C_matrix[..., c].detach().numpy(),
                aspect="auto",
                cmap="magma",
                origin="upper",
            )
            ax.set_title(f"{row_label}\nConcept {c}")
            ax.set_ylabel("Query")
            ax.set_xlabel("Sequence position")
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    def _plot_quiver_row(ax_row, C_matrix, disp, i_grid, j_grid, row_label, show_border=False):
        """Fill one row of axes with per-class quiver / displacement-field plots."""
        H, W, C = C_matrix.shape
        for c in range(C):
            ax = ax_row[c]
            slice_c = C_matrix[..., c].detach().numpy()
            ax.quiver(
                j_grid.numpy(),
                i_grid.numpy(),
                disp[..., c, 1].numpy(),
                -disp[..., c, 0].numpy(),
                slice_c,
                cmap="magma",
            )
            ax.invert_yaxis()
            ax.set_title(f"{row_label}\nConcept {c}")
            if show_border:
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(1.2)
                ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            else:
                ax.axis("off")

    # AW displayed as a heatmap — quiver is unsuitable for the [4, 1000] aspect ratio.
    AW = torch.stack(attn_contribution_matrices, dim=-1)
    QT = torch.stack(qt_contribution_matrices, dim=-1)
    qt_disp, qt_i, qt_j = _displacement_field(QT)

    # Plot: attn_weights as heatmaps (top row), query_token as quiver (bottom row)
    _fig, _axes = plt.subplots(2, embed_dim, figsize=(20, 8))
    _plot_attn_row(_axes[0], AW, row_label="Attn Weights")
    _plot_quiver_row(_axes[1], QT, qt_disp, qt_i, qt_j, row_label="Query Token", show_border=True)
    _fig.suptitle("Exploded Representation — Attention × Gradient", fontsize=14)
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
