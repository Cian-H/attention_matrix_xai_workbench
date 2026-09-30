import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def overview(mo):
    mo.md(r"""
    This experiment aims to try and use contribution sorted displacement fields to "explode"
    the entangled representations of the concepts in the attention matrices out into
    disentangled representations in a higher dimensional space. The basic algorithm would be
    summarised as follows:

    1. Take the attention matrix (`query_token` in this model) - example shape: (8, 12)

    2. Get an identically shaped array for each output class, scoring the contribution of the
    numbers in the attention matrices to each class. Then these with the original attention matrix
    to give a "contribution matrix". e.g: for 4 class classification this would give us a
    contribution matrix with a shape of (8, 12, 5)

    3. Along dimension 2, we sort each (8, 12) slice and calculate an (8, 12) displacement field
    along each axis and stack them. This results in an (8, 12, 2) displacement matrix that
    separates the axes. Then we re-stack those along axis 2 resulting in an (8, 12, 5, 2) matrix.
    I guess i would call this an "attention contribution displacement matrix"? The visualisation
    of it would look like an explosion, where every contribution maps onto a unique point in the
    displacement space though, so i'm just going to refer to it as the "exploded representation"
    for now.

    This exploded representation should provide a unique fingerprint for how each value in the
    original attention matrix contributes to the learned representations for each concept/class.
    The only question then becomes: how do we take that high dimensional, exploded view and
    collapse it back down to something human interpretable?
    """)
    return


@app.cell
def import_dependencies():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from sklearn.decomposition import PCA
    from torch import nn
    from torch.utils.data import DataLoader

    from attention_matrix_xai_workbench.attribution import attention_x_gradient
    from attention_matrix_xai_workbench.data import CausalSignalDataset, CausalSignalGenerator
    from attention_matrix_xai_workbench.model import MultiheadPooledAttention

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return (
        CausalSignalDataset,
        CausalSignalGenerator,
        DataLoader,
        MultiheadPooledAttention,
        PCA,
        attention_x_gradient,
        device,
        mo,
        nn,
        np,
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
def compute_attributions(
    PCA,
    attention_x_gradient,
    device,
    loader,
    np,
    schema,
    torch,
    trained_model,
):
    # Grab a single batch and slice out one observation
    _xb, _yb = next(iter(loader))
    _xb = _xb.to(device)
    x_single = _xb[0:1]

    embed_dim = len(schema)
    qt_contribution_matrices = []

    for class_idx in range(embed_dim):
        contrib = attention_x_gradient(
            trained_model,
            x_single,
            target_out_idx=class_idx,
            attention_layer_name="attn_matrix_hookpoint",
        )
        qt_contribution_matrices.append(contrib.query_token[0])

    QT = torch.stack(qt_contribution_matrices, dim=-1)
    T, D, C = QT.shape
    exploded_5d = QT.view(T * D, C).detach().cpu().numpy()

    pca = PCA(n_components=2)
    pca_2d = pca.fit_transform(exploded_5d)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    concept_descriptions = {
        "A": "A (sine)",
        "B": "B (square)",
        "C": "C (dep A)",
        "D": "D (dep B)",
        "E": "E (dep A+B)",
    }
    concept_labels = [concept_descriptions.get(k, str(k)) for k in schema.keys()]

    return (
        C,
        D,
        T,
        concept_labels,
        exploded_5d,
        loadings,
        pca,
        pca_2d,
    )


@app.cell
def plot_raw_pca(C, concept_labels, exploded_5d, np, pca, pca_2d, plt):
    _fig, _axes = plt.subplots(1, C, figsize=(22, 4.2))
    _pc1_min, _pc1_max = pca_2d[:, 0].min(), pca_2d[:, 0].max()
    _pc2_min, _pc2_max = pca_2d[:, 1].min(), pca_2d[:, 1].max()
    _p1 = (_pc1_max - _pc1_min) * 0.12 if _pc1_max > _pc1_min else 0.1
    _p2 = (_pc2_max - _pc2_min) * 0.12 if _pc2_max > _pc2_min else 0.1
    _exp_var = pca.explained_variance_ratio_

    for _c, _label in enumerate(concept_labels):
        _ax = _axes[_c]
        _colors = exploded_5d[:, _c]
        _sort_idx = np.argsort(np.abs(_colors))
        _vmax = max(float(np.abs(_colors).max()), 1e-6)

        _sc = _ax.scatter(
            pca_2d[_sort_idx, 0],
            pca_2d[_sort_idx, 1],
            c=_colors[_sort_idx],
            cmap="coolwarm",
            vmin=-_vmax,
            vmax=_vmax,
            s=45,
            alpha=0.9,
            edgecolors="black",
            linewidths=0.5,
        )
        _ax.set_xlim(_pc1_min - _p1, _pc1_max + _p1)
        _ax.set_ylim(_pc2_min - _p2, _pc2_max + _p2)
        _ax.set_title(f"Query Token PCA\n{_label}", fontsize=11, fontweight="bold")
        _ax.set_xlabel(f"PC1 ({_exp_var[0] * 100:.1f}%)", fontsize=9)
        if _c == 0:
            _ax.set_ylabel(f"PC2 ({_exp_var[1] * 100:.1f}%)", fontsize=9)
        else:
            _ax.tick_params(labelleft=False)

        _ax.grid(True, linestyle=":", alpha=0.5)
        _cbar = plt.colorbar(_sc, ax=_ax, fraction=0.046, pad=0.03)
        _cbar.formatter.set_powerlimits((-2, 2))
        _cbar.ax.yaxis.set_offset_position("left")
        _cbar.ax.tick_params(labelsize=8)

    _fig.suptitle(
        "Under-the-Hood: Raw Query Token PCA Projections per Concept",
        fontsize=13,
        y=1.02,
    )
    _fig.tight_layout()
    _fig
    return


@app.cell
def plot_cosine_similarity_cell(concept_labels, exploded_5d, np, plt):
    _fig, _ax = plt.subplots(figsize=(7, 6))
    _norms = np.linalg.norm(exploded_5d, axis=0, keepdims=True) + 1e-9
    _cos_sim = (exploded_5d / _norms).T @ (exploded_5d / _norms)
    _im = _ax.imshow(_cos_sim, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    _ax.set_xticks(range(len(concept_labels)))
    _ax.set_yticks(range(len(concept_labels)))
    _ax.set_xticklabels(concept_labels, rotation=30, ha="right", fontsize=9)
    _ax.set_yticklabels(concept_labels, fontsize=9)
    _ax.set_title(
        "1. Concept Attribution Cosine Similarity\n(Pairwise Alignment Across Parameters)",
        fontsize=11,
        fontweight="bold",
    )
    _cbar = plt.colorbar(_im, ax=_ax, fraction=0.046, pad=0.04)
    _cbar.ax.tick_params(labelsize=8)

    for _i in range(len(concept_labels)):
        for _j in range(len(concept_labels)):
            _val = _cos_sim[_i, _j]
            _ax.text(
                _j,
                _i,
                f"{_val:.2f}",
                ha="center",
                va="center",
                color="black" if abs(_val) < 0.6 else "white",
                fontsize=9.5,
                fontweight="bold",
            )

    _fig.tight_layout()
    _fig
    return


@app.cell
def plot_pca_biplot_cell(D, T, concept_labels, loadings, np, pca, pca_2d, plt):
    _fig, _ax = plt.subplots(figsize=(8, 7))
    _query_indices = np.repeat(np.arange(T), D)
    _palette = plt.cm.tab10(np.linspace(0, 1, 10))

    for _t in range(T):
        _mask = _query_indices == _t
        _ax.scatter(
            pca_2d[_mask, 0],
            pca_2d[_mask, 1],
            label=f"Query {_t}",
            s=60,
            alpha=0.85,
            color=_palette[_t],
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
        )

    _max_pca_x = np.max(np.abs(pca_2d[:, 0]))
    _max_pca_y = np.max(np.abs(pca_2d[:, 1]))
    _max_loading_x = np.max(np.abs(loadings[:, 0])) + 1e-9
    _max_loading_y = np.max(np.abs(loadings[:, 1])) + 1e-9
    _loading_scale = min(
        (_max_pca_x / _max_loading_x) * 0.75,
        (_max_pca_y / _max_loading_y) * 0.75,
    )

    _arrow_coords = []
    for _c, _label in enumerate(concept_labels):
        _dx = loadings[_c, 0] * _loading_scale
        _dy = loadings[_c, 1] * _loading_scale
        _arrow_coords.append((_dx, _dy))
        _ax.annotate(
            "",
            xy=(_dx, _dy),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="crimson", lw=2.0, mutation_scale=14),
            zorder=4,
        )
        _ax.text(
            _dx * 1.18,
            _dy * 1.18,
            _label,
            color="darkred",
            fontweight="bold",
            fontsize=9,
            ha="center",
            va="center",
            zorder=5,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor="none"),
        )

    _all_x = list(pca_2d[:, 0]) + [_x * 1.35 for _x, _ in _arrow_coords]
    _all_y = list(pca_2d[:, 1]) + [_y * 1.35 for _, _y in _arrow_coords]
    _p_x = (max(_all_x) - min(_all_x)) * 0.12
    _p_y = (max(_all_y) - min(_all_y)) * 0.12
    _ax.set_xlim(min(_all_x) - _p_x, max(_all_x) + _p_x)
    _ax.set_ylim(min(_all_y) - _p_y, max(_all_y) + _p_y)

    _exp_var = pca.explained_variance_ratio_
    _ax.axhline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.7, zorder=1)
    _ax.axvline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.7, zorder=1)
    _ax.grid(True, linestyle=":", alpha=0.5, zorder=0)
    _ax.set_xlabel(f"PC1 ({_exp_var[0] * 100:.1f}%)", fontsize=9)
    _ax.set_ylabel(f"PC2 ({_exp_var[1] * 100:.1f}%)", fontsize=9)
    _ax.set_title(
        "2. Unified PCA Biplot\n(Concepts as Loadings + Query Tokens)",
        fontsize=11,
        fontweight="bold",
    )
    _ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    _fig.tight_layout()
    _fig
    return


@app.cell
def plot_token_specialization_cell(D, T, concept_labels, exploded_5d, np, plt):
    _fig, _ax = plt.subplots(figsize=(7, 5))
    _token_contrib = np.zeros((T, len(concept_labels)))
    for _t in range(T):
        _token_contrib[_t, :] = np.linalg.norm(exploded_5d[_t * D : (_t + 1) * D, :], axis=0)

    _im = _ax.imshow(_token_contrib, cmap="viridis")
    _ax.set_xticks(range(len(concept_labels)))
    _ax.set_yticks(range(T))
    _ax.set_xticklabels(concept_labels, rotation=30, ha="right", fontsize=9)
    _ax.set_yticklabels([f"Query {_t}" for _t in range(T)], fontsize=9)
    _ax.set_title(
        "3. Query Token Specialization Matrix\n(Attribution Magnitude per Latent Slot)",
        fontsize=11,
        fontweight="bold",
    )
    _cbar = plt.colorbar(_im, ax=_ax, fraction=0.046, pad=0.04)
    _cbar.ax.tick_params(labelsize=8)

    _max_val = _token_contrib.max()
    for _t in range(T):
        for _c in range(len(concept_labels)):
            _val = _token_contrib[_t, _c]
            _fmt_val = f"{_val:.2e}" if _max_val < 0.01 else f"{_val:.3f}"
            _ax.text(
                _c,
                _t,
                _fmt_val,
                ha="center",
                va="center",
                color="white" if _val > _max_val * 0.55 else "black",
                fontsize=9,
                fontweight="bold",
            )

    _fig.tight_layout()
    _fig
    return


@app.cell
def plot_feature_fingerprints_cell(D, T, concept_labels, exploded_5d, np, plt):
    _fig, _axes = plt.subplots(T, D, figsize=(20, 10))
    _max_abs = float(np.abs(exploded_5d).max())
    _vmax = _max_abs * 1.15 if _max_abs > 0 else 1.0
    _c_short = [c.split()[0] for c in concept_labels]
    _colors = plt.cm.tab10(np.linspace(0, 0.5, len(_c_short)))

    for _t in range(T):
        for _d in range(D):
            _ax = _axes[_t, _d]
            _v = exploded_5d[_t * D + _d, :]
            _spec = np.max(np.abs(_v)) / (np.sum(np.abs(_v)) + 1e-9)

            _ax.bar(range(len(_c_short)), _v, color=_colors, alpha=0.85, width=0.65)
            _ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.7)
            _ax.set_ylim(-_vmax, _vmax)
            _ax.grid(True, linestyle=":", alpha=0.35, axis="y")

            _title = f"Q{_t}-D{_d} (s:{_spec:.2f})"
            if _t == 0:
                _ax.set_title(f"Dim {_d}\n{_title}", fontsize=8.5, fontweight="bold")
            else:
                _ax.set_title(_title, fontsize=8)

            if _d == 0:
                _ax.set_ylabel(f"Query {_t}", fontsize=9, fontweight="bold")
                _ax.ticklabel_format(style="sci", scilimits=(-2, 2), axis="y")
                _ax.tick_params(axis="y", labelsize=7)
            else:
                _ax.tick_params(labelleft=False)

            if _t == T - 1:
                _ax.set_xticks(range(len(_c_short)))
                _ax.set_xticklabels(_c_short, fontsize=8, fontweight="bold")
            else:
                _ax.tick_params(labelbottom=False)

    _fig.suptitle(
        "4. Latent Parameter Attribution Fingerprints "
        "(Query Token × Embedding Dim → Concept Routing)",
        fontsize=13,
        y=1.01,
        fontweight="bold",
    )
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
