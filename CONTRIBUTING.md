# Contributing to Hupesi TTS

Thank you for your interest. All contributions are welcome: bug reports, fixes,
UI improvements, documentation, and new features.

## Before you start

- **Small bugs, typos, documentation:** open a pull request directly.
- **New features or large changes:** open an issue first to align on the
  approach, so you don't spend time on code that won't be accepted.
- **Security vulnerabilities:** do not open a public issue; see
  [SECURITY.md](SECURITY.md).

## Development setup

Requirements: Windows, macOS, or Linux; [uv](https://docs.astral.sh/uv/);
Node.js 20 or later.

```bash
# Python 3.12 and dependencies (uv installs Python automatically if needed)
uv sync --group desktop --group dev

# React UI
cd app-desktop/hupesi-tts-app
npm ci
npm run build        # creates dist/ for the backend to serve the UI
cd ../..

# Run the app
uv run python -m apps.desktop_app
```

On the first run, the app downloads about 312 MB of models to `models/hf_cache`.
To download them in advance: `uv run python scripts/fetch_models.py`.

When working on the UI, run `npm run dev` inside `app-desktop/hupesi-tts-app`
(port 1420) alongside the backend to see changes immediately.

Architecture notes and common pitfalls are documented in code comments (at the
top of each module in `apps/` and `src/`). Read them before modifying core
components.

## Repository structure

| Directory | Contents |
|---|---|
| `apps/` | FastAPI server (`web_stream.py`), desktop window (`desktop_app.py`), model management, Supertonic, Kokoro |
| `src/vieneu/` | VieNeu-TTS v3 Turbo / Nano engine (from upstream VieNeu-TTS) |
| `src/vieneu_utils/` | Text normalization, sentence splitting, subtitles, pause tags, pronunciation dictionary |
| `app-desktop/hupesi-tts-app/` | React UI |
| `scripts/` | Packaging and release (PyInstaller, Inno Setup, .deb, .dmg) |
| `packaging/` | OS-specific installer assets |
| `tests/` | Python tests (pytest) |

## Checks before opening a pull request

```bash
uv run pytest                                  # all Python tests
cd app-desktop/hupesi-tts-app
npm run check                                  # Biome lint (errors block; warnings should be fixed)
npm run typecheck
npm test
npm run build
```

Tests do not download real models, so they run quickly and do not require a
network connection.

## Conventions

- **Language:** comments, logs, and all user-facing text should be in English
  (or follow existing project conventions where applicable); variable, function,
  and API names are in English.
- **Core must not require PyTorch:** the `vieneu` package must run on CPU
  without torch. Do not `import torch`, `transformers`, `librosa`, etc. at the
  top level of modules on the CPU path; place them inside functions.
- **Licensing:** the project is Apache-2.0. Do not bundle GPL libraries in the
  installer. GPL libraries (such as espeak-ng for Kokoro) may only be installed
  on the user's machine when they click Download. When adding a new bundled
  library, update [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).
- **Network access:** the app runs offline. Any new external connection must be
  documented in [PRIVACY.md](PRIVACY.md).
- **Exported audio:** every file export path must add the "AI-generated" label
  (see `apps/ai_label.py`).
- **API:** endpoints used only by the UI should be declared with
  `include_in_schema=False`. Public endpoints must also update
  `docs/hupesi-tts.md`.
- **Commits:** use forms such as `feat: ...`, `fix: ...`, `docs: ...`, with a
  short description in Vietnamese or English.

## Pull requests

- Each pull request should address one thing.
- Describe: what changed, why, and how it was tested. Include screenshots for UI
  changes.
- Add or update tests for the logic you change.

By submitting a contribution, you agree that your contribution will be released
under the project's [Apache-2.0](LICENSE) license.