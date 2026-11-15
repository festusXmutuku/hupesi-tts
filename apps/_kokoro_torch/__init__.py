"""PyTorch Kokoro-82M (copied from hexgrad/kokoro 0.9.4, Apache-2.0) for the GPU path.

Model only (`KModel`): phonemization is still handled by `apps/kokoro_engine.py`. The full
`kokoro` package is not installed because it requires `misaki[en]` (spaCy + English model)
even though this path does not use it. Imports torch/transformers - only load after the GPU
venv has been added to sys.path (see `apps/gpu_installer.activate_gpu_venv`).
"""
