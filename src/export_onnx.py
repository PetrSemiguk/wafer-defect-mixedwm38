import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from dataset import WaferDataset
from model import WaferCNN


def benchmark(fn, x, repeats):
    """Median and 95th percentile time per call, in milliseconds."""
    for _ in range(10):  # warm-up
        fn(x)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn(x)
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times)), float(np.percentile(times, 95))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/wafers_clean.npz")
    parser.add_argument("--ckpt", default="runs/exp01/best.pt")
    parser.add_argument("--onnx", default="models/wafer_cnn.onnx")
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()

    model = WaferCNN()
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
    model.eval()

    # --- 1. Export with a dynamic batch dimension ---
    onnx_path = Path(args.onnx)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, (torch.zeros(1, 3, 52, 52),), str(onnx_path),
        input_names=["wafer"], output_names=["logits"],
        dynamic_axes={"wafer": {0: "batch"}, "logits": {0: "batch"}},
        external_data=False,  # keep weights inside the .onnx file
    )
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    print(f"Exported {onnx_path} ({onnx_path.stat().st_size / 1e6:.2f} MB)")

    # --- 2. Check that ONNX Runtime reproduces PyTorch on real test wafers ---
    ds = WaferDataset(args.data, "test")
    x = torch.stack([ds[i][0] for i in range(512)])
    with torch.no_grad():
        ref = model(x).numpy()
    out = session.run(None, {"wafer": x.numpy()})[0]
    max_diff = float(np.abs(ref - out).max())
    same_labels = bool(((ref > 0) == (out > 0)).all())
    print(f"Max logit difference: {max_diff:.2e} | identical predictions on 512 wafers: {same_labels}")

    # --- 3. CPU latency: single wafer and batch of 64 ---
    def run_torch(batch):
        with torch.no_grad():
            model(batch)

    def run_onnx(batch):
        session.run(None, {"wafer": batch})

    print(f"\nCPU: {platform.processor() or platform.machine()} | torch threads: {torch.get_num_threads()}")
    print(f"{'Runtime':<14} {'Batch':>5} {'Median ms':>10} {'p95 ms':>8} {'Wafers/s':>9}")
    results = []
    for batch_size, repeats in [(1, 300), (64, 30)]:
        batch = x[:batch_size]
        for name, fn, data in [("PyTorch", run_torch, batch), ("ONNX Runtime", run_onnx, batch.numpy())]:
            median, p95 = benchmark(fn, data, repeats)
            throughput = batch_size / median * 1000
            results.append({"runtime": name, "batch": batch_size, "median_ms": median,
                            "p95_ms": p95, "wafers_per_s": throughput})
            print(f"{name:<14} {batch_size:>5} {median:>10.2f} {p95:>8.2f} {throughput:>9.0f}")

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    json.dump(
        {"cpu": platform.processor(), "onnx_size_mb": onnx_path.stat().st_size / 1e6,
         "max_logit_diff": max_diff, "identical_predictions": same_labels, "results": results},
        open(out_dir / "latency.json", "w"), indent=2,
    )
    print(f"\nSaved {out_dir / 'latency.json'}")


if __name__ == "__main__":
    main()