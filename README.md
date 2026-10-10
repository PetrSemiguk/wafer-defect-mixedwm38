# Mixed-Type Wafer Map Defect Recognition (MixedWM38)

Multi-label CNN that detects 8 defect types on semiconductor wafer maps,
including wafers with several overlapping defects.

**Status:** work in progress.

## Results so far (held-out test set, 5,549 wafers)

- Macro-F1: 0.991
- Exact match (all 8 labels correct): 98.4%
- Per-defect metrics, error analysis and threshold trade-offs: see `reports/`

## Next

- Comparison with a 38-class formulation
- Grad-CAM visualizations
- ONNX export and CPU latency
- Web demo