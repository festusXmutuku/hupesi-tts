# On-Demand GPU Support Package

The desktop build only bundles `tools/uv.exe`; it does not include PyTorch,
torchaudio, or transformers. Build with `scripts/build_desktop_exe.bat` to
include this tool in the bundle. Users on older `.exe` builds must update the
app to receive this change.

In Settings, an NVIDIA machine that does not yet have the GPU libraries will
show **Install GPU support**. This button downloads the model and installs the
libraries in a background worker, even if the model is already available. CPU
machines do not download the GPU libraries. Backends running with a separate
Python installation can also install from Settings when the `uv` tool is found;
the PyInstaller flag is not required.

The first installation requires a network connection and at least 12 GB of free
space. Python is selected to match the app's major/minor version; data is stored
in `gpu_venv/<Python version>` next to the app, or in
`%LOCALAPPDATA%/HupesiTTS/gpu_venv` if the app directory is read-only.
Download and Python caches are shared in the parent directory. PyTorch 2.8.0 and
torchaudio 2.8.0 are fetched from the CUDA cu128 index; transformers 4.57.6 and
safetensors are fetched from PyPI.

The interface shows each installation step; it does not use model download
percentages to represent the size of the GPU package. Cancel stops the
installation process; Retry reuses the downloaded cache. The app verifies
imports and CUDA operations in the separate Python environment and then in the
app process before applying the model. If only the in-app loading step fails,
the user is guided to restart so the libraries are loaded from the beginning.

## Release verification on Windows with NVIDIA

1. Build a new `.exe` and confirm that `_internal/tools/uv.exe` exists. Verify
   that torch/CUDA are not present in the CPU bundle.
2. On an RTX 3060 machine with a compatible driver, without Python/uv in PATH,
   open Settings and click Install GPU support. Confirm that the installation
   steps are displayed clearly.
3. Wait for the installation to finish, confirm that GPU is applied, and
   synthesize a test sentence.
4. Reopen the app: it should reuse the installed libraries without downloading
   the GPU package again.
5. Test cancellation during installation, network failure followed by Retry,
   insufficient disk space, a read-only installation directory, and a case where
   the model is available but GPU libraries are not installed yet.
6. On a CPU machine, open the app and use the int8 build: it must not download
   the GPU support package.

Unit tests use simulated GPU and network conditions; they do not replace testing
the `.exe` on a real RTX 3060.