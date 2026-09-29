import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def import_dependencies():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from attention_matrix_xai_workbench.data import CausalSignalDataset, CausalSignalGenerator
    from attention_matrix_xai_workbench.model import MultiheadPooledAttention

    return (
        CausalSignalDataset,
        CausalSignalGenerator,
        DataLoader,
        MultiheadPooledAttention,
        mo,
        nn,
        plt,
        torch,
    )


@app.cell
def create_schema():
    # Simple schema for demonstration, generates a sine wave feature `a` and a dependent feature `b`
    schema = {
        "signal_a": {"type": "source", "waveform": "sine", "freq_range": (5, 15)},
        "signal_b": {"type": "mix", "depends_on": {"signal_a": "ramp_up"}, "interaction": "add"},
    }
    return (schema,)


@app.cell
def device(torch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return (device,)


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
    model = MultiheadPooledAttention(embed_dim=2).float().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    return criterion, model, optimizer


@app.cell
def train(criterion, dataset, device, loader, mo, model, optimizer, plt):
    losses: list[float] = []
    fig, ax = plt.subplots(figsize=(6, 4))

    for epoch in mo.status.progress_bar(range(100), title="Training Epochs"):
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
    return


if __name__ == "__main__":
    app.run()
