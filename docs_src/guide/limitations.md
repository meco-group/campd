# Limitations

- **CUDA GPU required.** The framework is not tested on CPU-only or MPS (Apple Silicon) setups.
- **Windows is not natively supported.** Use Linux, macOS, or WSL2 on Windows.
- **Metrics and visualizations are not built-in.** Training loss is logged to the console and optionally to W&B. Inference prints a timing and stats summary. Domain-specific evaluation (collision rate, trajectory quality, plots) must be implemented as a custom `Validator` or `Summary` for your project. See [Extending the Framework](extending.md).
- **CUDA graph training requires fixed tensor shapes.** If `cuda_graph.enabled: true`, the batch size, trajectory length, state dimension, and context sizes must be constant across all batches. CUDA graphs are also incompatible with TorchJD multi-objective optimization and Accelerate DDP.
- **The framework targets trajectory-shaped data** (fixed-length sequences stored in HDF5). Variable-length outputs or non-sequential data require adapting the data layer.
