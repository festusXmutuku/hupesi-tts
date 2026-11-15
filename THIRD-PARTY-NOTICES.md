# Third-Party Notices - Hupesi TTS

Hupesi TTS uses the open-source components listed below. Each component
retains its original copyright and license. Attribution for VieNeu-TTS and for
components downloaded only when the user opts in (GPU support, optional models)
is provided in the `NOTICE` file.

## LGPL libraries

These libraries are used **unmodified** as dynamically linked libraries located
separately in the installation directory. You have the right to replace them
with compatible alternatives.

| Component | License | Source |
|---|---|---|
| libsndfile (read/write WAV, MP3) | LGPL-2.1-or-later | https://github.com/libsndfile/libsndfile |
| LAME (MP3 encoder, in libsndfile) | LGPL-2.0-or-later | https://lame.sourceforge.io |
| mpg123 (MP3 decoder, in libsndfile) | LGPL-2.1 | https://www.mpg123.de |
| libsoxr (sample-rate conversion, via `soxr`) | LGPL-2.1-or-later | https://github.com/dofuuz/python-soxr |

The full LGPL-2.1 text is included in the installation directory in
`_internal/_soundfile_data/COPYING` (Windows builds).

## Bundled AI models

| Model | License | Source |
|---|---|---|
| VieNeu-TTS v3 Turbo (ONNX int8) and bundled preset voices | Apache-2.0 | https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Turbo |
| MOSS Audio Tokenizer Nano (ONNX) | Apache-2.0 | https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX |

## Models downloaded on demand

| Model | License | Source |
|---|---|---|
| Supertonic 3 (Supertone Inc.) - foreign-language voices for 31 languages, downloaded when you click Download in Models. The license `LICENSE` file is downloaded with the model. The license includes usage restrictions (no impersonation without consent, fraud, harmful misinformation, unlabeled machine-generated content, etc.) that apply to all users. | OpenRAIL-M ([LICENSE](https://huggingface.co/supertone-oss-archive/supertonic-3/blob/main/LICENSE)) | https://huggingface.co/supertone-oss-archive/supertonic-3 (sample code, MIT: https://github.com/supertone-oss-archive/supertonic, [LICENSE](https://github.com/supertone-oss-archive/supertonic/blob/main/LICENSE)) |
| Kokoro-82M (hexgrad) - foreign-language voices for 8 language groups, downloaded when you click Download in Models (CPU uses the ONNX export; GPU uses the original PyTorch weights) | Apache-2.0 | https://huggingface.co/hexgrad/Kokoro-82M, https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX |

## Kokoro: model runner code and phonemizer

The Kokoro GPU runner code (`apps/_kokoro_torch`) is copied from
hexgrad/kokoro 0.9.4, Apache-2.0, and includes the original LICENSE file.
Kokoro's phonemizer is **not included in the installer**: when you click
Download, your machine downloads it from PyPI into a separate directory
(`addons/`). This includes `phonemizer` (GPL-3.0-or-later) and espeak-ng via
`espeakng-loader` (GPL-3.0); the full package list and licenses are recorded in
`NOTICE`.

## Other libraries

The application also uses approximately 90 open-source libraries under MIT, BSD,
Apache-2.0, ISC, PSF, MPL-2.0, and SIL Open Font License 1.1, including:
Python, NumPy, SciPy, ONNX Runtime, Hugging Face Hub, FastAPI, Uvicorn,
pywebview, pythonnet, OpenSSL, PyInstaller (bootloader under GPL-2.0 with an
exception allowing distribution), React, Radix UI, Lucide, Tailwind CSS, and the
Inter and Be Vietnam Pro fonts. Copyright belongs to the authors of each
library.