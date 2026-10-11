import json
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.figure import Figure

from model import WaferCNN

ROOT = Path(__file__).parent
DEFECTS = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Scratch", "Random"]
ANY = "Any combination"
BALANCED = "Balanced (threshold 0.5)"
SENSITIVE = "High sensitivity (tuned thresholds)"

# --- Load sample wafers, thresholds and the model once at startup ---
data = np.load(ROOT / "samples.npz")
MAPS, LABELS = data["maps"], data["labels"].astype(int)
NAMES = np.array(["+".join(d for d, flag in zip(DEFECTS, row) if flag) or "Normal" for row in LABELS])
tuned = json.load(open(ROOT / "thresholds.json"))
THRESHOLDS = {BALANCED: np.full(8, 0.5), SENSITIVE: np.array([tuned[d] for d in DEFECTS])}

model = WaferCNN()
model.load_state_dict(torch.load(ROOT / "wafer_cnn_weights.pth", map_location="cpu"))
model.eval()


def analyze(wafer):
    """Probabilities for all 8 defects and one Grad-CAM heatmap per defect."""
    x = F.one_hot(torch.from_numpy(wafer).long(), num_classes=3).permute(2, 0, 1).float()[None]
    features = model.features(x)
    logits = model.head(features)[0]
    cams = []
    for k in range(len(DEFECTS)):
        grads = torch.autograd.grad(logits[k], features, retain_graph=True)[0]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * features).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=wafer.shape, mode="bilinear", align_corners=False)[0, 0]
        cams.append((cam / (cam.max() + 1e-8)).detach().numpy())
    return torch.sigmoid(logits).detach().numpy(), cams


def pick(combo):
    """Index of a random sample wafer, optionally restricted to one combination."""
    candidates = np.arange(len(MAPS)) if combo == ANY else np.where(NAMES == combo)[0]
    return int(np.random.choice(candidates))


def render(idx, mode):
    wafer, truth = MAPS[idx], LABELS[idx]
    probs, cams = analyze(wafer)
    thresholds = THRESHOLDS[mode]
    pred = (probs >= thresholds).astype(int)

    def status(k):
        if pred[k] == truth[k]:
            return "correct"
        return "MISSED" if truth[k] else "FALSE ALARM"

    rows = [
        [name, f"{probs[k]:.3f}", f"{thresholds[k]:.2f}",
         "yes" if pred[k] else "no", "yes" if truth[k] else "no", status(k)]
        for k, name in enumerate(DEFECTS)
    ]
    pred_name = "+".join(d for d, flag in zip(DEFECTS, pred) if flag) or "Normal"
    verdict = "all 8 labels correct" if (pred == truth).all() else "at least one label wrong"
    summary = f"**True label:** {NAMES[idx]}  \n**Prediction:** {pred_name}  \n**Result:** {verdict}"

    # Wafer map plus one heatmap for every defect that is predicted or actually present
    shown = [k for k in range(len(DEFECTS)) if pred[k] or truth[k]]
    fig = Figure(figsize=(3.2 * (len(shown) + 1), 3.7))
    axes = fig.subplots(1, len(shown) + 1, squeeze=False)[0]
    axes[0].imshow(wafer, cmap="gray", vmin=0, vmax=2)
    axes[0].set_title("Wafer map\n(white = failed die)", fontsize=10)
    for ax, k in zip(axes[1:], shown):
        ax.imshow(wafer, cmap="gray", vmin=0, vmax=2)
        ax.imshow(cams[k], cmap="jet", alpha=0.5, vmin=0, vmax=1)
        ax.set_title(f"{DEFECTS[k]}  p={probs[k]:.2f}\n{status(k)}", fontsize=10)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    return summary, rows, fig


with gr.Blocks(title="Wafer Defect Recognition") as demo:
    gr.Markdown(
        "# Mixed-Type Wafer Defect Recognition\n"
        "A multi-label CNN detects 8 defect types on wafer maps from the MixedWM38 test set. "
        "Pick a wafer to see the prediction, the ground truth and a Grad-CAM heatmap for each defect. "
        "Switch the operating mode to trade missed defects against false alarms."
    )
    idx = gr.State(0)
    with gr.Row():
        combo = gr.Dropdown([ANY] + sorted(set(NAMES.tolist())), value=ANY, label="Defect combination")
        mode = gr.Radio([BALANCED, SENSITIVE], value=BALANCED, label="Operating mode")
        button = gr.Button("Random wafer", variant="primary")
    summary = gr.Markdown()
    plot = gr.Plot(label="Wafer map and Grad-CAM per defect")
    table = gr.Dataframe(
        headers=["Defect", "Probability", "Threshold", "Predicted", "Actual", "Result"],
        interactive=False,
    )
    outputs = [summary, table, plot]

    button.click(pick, combo, idx).then(render, [idx, mode], outputs)
    combo.change(pick, combo, idx).then(render, [idx, mode], outputs)
    mode.change(render, [idx, mode], outputs)
    demo.load(pick, combo, idx).then(render, [idx, mode], outputs)

if __name__ == "__main__":
    demo.launch()