# Troubleshooting

## CUDA Graph Errors
If `use_cuda_graph: true` is configured in your training loop but execution rapidly crashes out with runtime errors:
1. Ensure your PyTorch CUDA version explicitly matches your system installed CUDA version architecture.
2. **Verify Batch Uniformity:** CUDA Graphs execute by recording an exact sequence of static memory addresses and tensor dimensions. If *any* batch you feed to the model suddenly changes in length, sequence size, or dimension size (e.g. an unsorted jagged-array DataLoader ending in a batch of `size=4` instead of `size=64`), the CUDA graph execution boundary will dynamically rupture and throw a shape mismatch error.

## `KeyError: Unknown 'X' in registry 'Y'`
This denotes that your component `X` has failed to load into the registry. Proceed with the following checks:
1. Is the string perfectly identically spelled in your decorator and YAML config?
2. Ensure the `@REGISTRY.register("X")` decorator is located *above* your class definition.
3. Most commonly, **the script file has not been imported**. If relying upon `campd-run`, ensure the directory or relative path locating your custom module is populated in the `dependencies` array within your config.
