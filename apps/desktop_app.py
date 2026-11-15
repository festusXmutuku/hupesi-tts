"""
Hupesi TTS - Desktop App Launcher (PyWebView + FastAPI + VieNeu-TTS)
========================================================================
Automatically starts the AI backend in the background and opens the native desktop application window.
No Node.js or Rust installation is required to run or package the app.
"""
import io
import json
import os
import socket
import sys
import time
import threading
from pathlib import Path
import urllib.request
import uvicorn

# PyWebView on Windows runs via pythonnet + .NET Framework. If the user's machine is missing
# .NET Framework 4.7.2+ (or the package is missing runtime files), this import will fail.
# In that case the app must not crash - it falls back to the default browser (see _run_in_browser).
try:
    import webview
    _WEBVIEW_IMPORT_ERROR = None
except Exception as _webview_exc:
    webview = None
    _WEBVIEW_IMPORT_ERROR = _webview_exc

# Fix Windows / PyInstaller Windowed Unicode Encoding issues
if sys.platform == "win32":
    if sys.stdout is None:
        sys.stdout = io.StringIO()
    else:
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    if sys.stderr is None:
        sys.stderr = io.StringIO()
    else:
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _log_path() -> "Path":
    """Log file location. Next to the .exe if writable, otherwise %LOCALAPPDATA%."""
    try:
        if sys.platform == "darwin" and getattr(sys, "frozen", False):
            # Do not write logs inside the .app bundle (see apps/runtime_paths_macos.py).
            from apps.runtime_paths_macos import data_dir
            return data_dir() / "hupesi-tts.log"
        base = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) \
            else Path(__file__).resolve().parents[1]
        probe = base / ".write_test"
        probe.write_bytes(b"")
        probe.unlink()
        return base / "hupesi-tts.log"
    except Exception:
        d = Path(os.getenv("LOCALAPPDATA") or Path.home()) / "HupesiTTS"
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return d / "hupesi-tts.log"


LOG_FILE = _log_path()
try:  # start a fresh log file on each run to avoid unbounded growth
    LOG_FILE.write_text("", encoding="utf-8")
except Exception:
    pass


def log(msg: str):
    """Safe logging that avoids charmap UnicodeEncodeError crashes.

    ALSO WRITES TO FILE: packaged builds run with `--windowed` so there is no console; everything printed
    disappears. Without a log file, errors on the user's machine cannot be
    diagnosed - exactly the situation encountered with the blank screen issue.
    """
    try:
        print(msg)
    except Exception:
        try:
            print(msg.encode("ascii", errors="replace").decode("ascii"))
        except Exception:
            pass
    try:
        with open(LOG_FILE, "a", encoding="utf-8", errors="replace") as f:
            f.write(f"{time.strftime('%H:%M:%S')}  {msg}\n")
    except Exception:
        pass


# Add root and src directories to sys.path (supports frozen exe)
if getattr(sys, "frozen", False):
    ROOT_DIR = Path(sys._MEIPASS)
else:
    ROOT_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

# Shared Hugging Face cache for all 3 entry points. Must run BEFORE importing
# web_stream (it pulls in huggingface_hub, which reads HF_HOME at import time).
from apps.runtime_paths import setup_hf_home
setup_hf_home()

from apps.server_settings import setup_port
from apps import remote_access

PORT = setup_port()
BACKEND_URL = f"http://127.0.0.1:{PORT}"

# Error that kills the backend thread (model load failure, port in use, etc.). main() reads this
# to report clearly to the user instead of opening a blank window pointing to a non-existent server.
_BACKEND_ERROR = None


WEBVIEW2_DOWNLOAD_URL = "https://go.microsoft.com/fwlink/p/?LinkId=2124703"


def webview2_available() -> bool:
    """Check whether Microsoft Edge WebView2 Runtime is installed.

    Why check manually: pywebview silently falls back to MSHTML (IE11 engine) when
    WebView2 is not found - even when `gui="edgechromium"` is forced (see `is_chromium` in
    webview/platforms/winforms.py). It only logs a warning, and packaged
    `--windowed` builds have no console to read it. IE11 cannot run the React bundle (Vite
    builds target ES2020), so users get a BLANK WINDOW with no explanation.

    Windows 11 includes WebView2. Windows 10 may not - LTSC editions or machines that have never
    installed the new Edge may be missing it.

    Detection logic follows pywebview's approach for consistency, but reads the registry directly
    so it works even when pythonnet is broken.
    """
    if sys.platform != "win32":
        return True
    try:
        import winreg
    except Exception:
        return False

    def pv(hive, sub: str) -> str:
        try:
            with winreg.OpenKey(hive, sub) as k:
                return str(winreg.QueryValueEx(k, "pv")[0])
        except Exception:
            return "0"

    def as_tuple(s: str):
        try:
            return tuple(int(x) for x in s.split("."))
        except Exception:
            return (0,)

    keys = (
        "{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}",  # Official runtime
        "{2CD8A007-E189-409D-A2C8-9AF4EF3C72AA}",  # Beta
        "{0D50BFEC-CD6A-4F9A-964C-C7416E3ACB10}",  # Dev
        "{65C35B14-6C1D-4122-AC46-7148CC9D6497}",  # Canary
    )
    for key in keys:
        for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
            for prefix in (r"SOFTWARE\Microsoft", r"SOFTWARE\WOW6432Node\Microsoft"):
                if as_tuple(pv(hive, rf"{prefix}\EdgeUpdate\Clients\{key}")) >= (86, 0, 622, 0):
                    return True
    return False


# Desired window size. Only a preference - always clamped to fit the screen.
WINDOW_PREFERRED = (1120, 720)
WINDOW_MIN = (800, 500)
# Margin so the window doesn't sit flush against the work area edge.
WINDOW_MARGIN = 16


def fit_window_to_screen(work, preferred=WINDOW_PREFERRED, minimum=WINDOW_MIN):
    """Calculate window size and position to fit ANY screen.

    `work` is the work area in LOGICAL pixels (taskbar excluded): (x, y, width, height).
    Returns a dict that can be passed directly to `webview.create_window`.

    Why compute manually: without x/y, pywebview falls back to
    `FormStartPosition.CenterScreen` in WinForms, which centers against the FULL
    screen height instead of excluding the taskbar - so the window ends up lower than the true
    center of the usable area.

    Worse is the size issue. Measured on a 2560x1440 display at 175% scale: the work area
    is only 775 logical px high, while a hardcoded 800px window is taller than the available space,
    so the bottom gets hidden under the taskbar. Smaller screens or higher scaling make it
    worse.

    So every value here is clamped to `work`, including min_size: a `min_size` larger than
    the screen would be enforced by WinForms, causing the window to overflow just like before.
    """
    wx, wy, ww, wh = (int(v) for v in work)
    ww, wh = max(ww, 1), max(wh, 1)

    # min_size khong bao gio duoc vuot vung lam viec.
    min_w = min(int(minimum[0]), ww)
    min_h = min(int(minimum[1]), wh)

    # Kich thuoc: lay theo mong muon, nhung khong qua vung lam viec tru mep.
    w = min(int(preferred[0]), max(min_w, ww - WINDOW_MARGIN))
    h = min(int(preferred[1]), max(min_h, wh - WINDOW_MARGIN))
    w = max(min_w, min(w, ww))
    h = max(min_h, min(h, wh))

    # Can giua theo VUNG LAM VIEC, khong phai theo man hinh.
    x = wx + (ww - w) // 2
    y = wy + (wh - h) // 2
    return {"width": w, "height": h, "x": x, "y": y, "min_size": (min_w, min_h)}


def _primary_work_area_logical():
    """Primary screen work area in LOGICAL pixels. None if it cannot be read.

    pywebview expects logical pixel coordinates and multiplies by the window DPI itself, so here
    we must return logical pixels, not physical pixels.

    Common pitfall: `GetSystemMetrics` / `SPI_GETWORKAREA` return values depend on whether the process
    has declared DPI awareness. Without it, Windows already scales the values (i.e. they are
    LOGICAL). With it, they are PHYSICAL pixels. Dividing again incorrectly makes the
    window tiny on 4K displays. So we compare with the real resolution from
    EnumDisplaySettings to determine which state we're in.
    """
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32

        class RECT(ctypes.Structure):
            _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                        ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

        rect = RECT()
        if not user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0):
            return None
        reported_h = user32.GetSystemMetrics(1)
        if reported_h <= 0:
            return None

        class DEVMODE(ctypes.Structure):
            _fields_ = [("dmDeviceName", wintypes.WCHAR * 32),
                        ("dmSpecVersion", wintypes.WORD), ("dmDriverVersion", wintypes.WORD),
                        ("dmSize", wintypes.WORD), ("dmDriverExtra", wintypes.WORD),
                        ("dmFields", wintypes.DWORD), ("dmPositionX", ctypes.c_long),
                        ("dmPositionY", ctypes.c_long), ("dmDisplayOrientation", wintypes.DWORD),
                        ("dmDisplayFixedOutput", wintypes.DWORD), ("dmColor", ctypes.c_short),
                        ("dmDuplex", ctypes.c_short), ("dmYResolution", ctypes.c_short),
                        ("dmTTOption", ctypes.c_short), ("dmCollate", ctypes.c_short),
                        ("dmFormName", wintypes.WCHAR * 32), ("dmLogPixels", wintypes.WORD),
                        ("dmBitsPerPel", wintypes.DWORD), ("dmPelsWidth", wintypes.DWORD),
                        ("dmPelsHeight", wintypes.DWORD), ("dmDisplayFlags", wintypes.DWORD),
                        ("dmDisplayFrequency", wintypes.DWORD), ("dmICMMethod", wintypes.DWORD),
                        ("dmICMIntent", wintypes.DWORD), ("dmMediaType", wintypes.DWORD),
                        ("dmDitherType", wintypes.DWORD), ("dmReserved1", wintypes.DWORD),
                        ("dmReserved2", wintypes.DWORD), ("dmPanningWidth", wintypes.DWORD),
                        ("dmPanningHeight", wintypes.DWORD)]

        dm = DEVMODE()
        dm.dmSize = ctypes.sizeof(DEVMODE)
        scale = 1.0
        if user32.EnumDisplaySettingsW(None, -1, ctypes.byref(dm)) and dm.dmPelsHeight:
            # Bao cao trung do phan giai that => dang DPI-aware => so lieu la VAT LY.
            if abs(int(dm.dmPelsHeight) - int(reported_h)) <= 1:
                try:
                    dpi = user32.GetDpiForSystem()
                except Exception:
                    dpi = 96
                if dpi and dpi > 0:
                    scale = dpi / 96.0
        if scale <= 0:
            scale = 1.0

        return (int(rect.left / scale), int(rect.top / scale),
                int((rect.right - rect.left) / scale), int((rect.bottom - rect.top) / scale))
    except Exception as e:
        log(f"[Diagnostics] Could not read screen work area: {e}")
        return None


def splash_close() -> None:
    """Close the PyInstaller splash screen (`--splash`) if it is showing.

    This splash screen is drawn by the bootloader BEFORE Python starts, so it is the only thing
    that can cover the first few seconds while the bundle is extracted. The text on it is static
    in `assets/splash.png`: the CLI `--splash` mode does not enable a dynamic text area.

    Must be called on EVERY exit path, otherwise the splash screen may overlay the real window
    or hide error dialogs.

    Only import when the bootloader actually enabled the splash: Linux/macOS builds don't have
    `--splash` but `pyi_splash` is still bundled, and importing it there prints
    a `KeyError: '_PYI_SPLASH_IPC'` traceback to the log.
    """
    if "_PYI_SPLASH_IPC" not in os.environ:
        return
    try:
        import pyi_splash  # only exists in packaged builds with --splash
        pyi_splash.close()
    except Exception:
        pass


def _show_error(title: str, message: str) -> None:
    """Show an error so the user can actually see it.

    Packaged PyInstaller builds run with `--windowed` so there is no console - anything printed
    goes nowhere. We must use a native Windows dialog.
    """
    log(f"[Desktop App] {title}: {message}")
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(0, message, title, 0x10)
        except Exception:
            pass


def _probe_health(url: str, timeout: float = 1.0):
    """Call /health. Returns a dict if it is this app's backend, None otherwise.

    Also checks the JSON shape: port 8001 may be occupied by another app that still
    returns HTTP 200 - in that case it must not be treated as our backend.
    """
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"{url}/health", timeout=timeout) as res:
            if res.status != 200:
                return None
            data = json.loads(res.read().decode("utf-8", errors="replace"))
    except Exception:
        return None
    if isinstance(data, dict) and "engine" in data and "sample_rate" in data:
        return data
    return None


def _reserve_backend_socket(preferred: int, host: str = "127.0.0.1"):
    """Reserve a private listener so concurrent launches cannot claim the same port."""
    for port in range(preferred, min(preferred + 21, 65536)):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if sys.platform == "win32":
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            else:
                # Linux/macOS: without this flag, a port stays in TIME_WAIT after closing the app
                # for ~60 seconds, so reopening (including the Restart button) gets pushed to 8002. On
                # POSIX this flag does NOT allow two processes to listen on the same port.
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((host, port))
            listener.listen(128)
            return listener
        except OSError:
            listener.close()
    raise RuntimeError(f"No free port available near port {preferred}.")


def _reserve_listener(preferred: int):
    """0.0.0.0 when "Allow access from other devices" is enabled (Settings), otherwise local only.
    If binding to 0.0.0.0 fails (network policy, etc.), fall back to 127.0.0.1 so the app still runs."""
    host = remote_access.listen_host()
    if host != "127.0.0.1":
        try:
            listener = _reserve_backend_socket(preferred, host)
            remote_access.mark_listening(host)
            log(f"[Desktop App] Remote access enabled: listening on {host}:{listener.getsockname()[1]}")
            return listener
        except Exception as e:
            log(f"[Desktop App] Could not bind {host}, falling back to local access only: {e}")
    listener = _reserve_backend_socket(preferred)
    remote_access.mark_listening("127.0.0.1")
    return listener


def run_backend(listener):
    """Run this window's backend on its already-reserved socket."""
    global _BACKEND_ERROR
    try:
        from apps.web_stream import app
        log(f"[Desktop App] Starting Backend API at {BACKEND_URL}...")
        # ws="none": the app does not use WebSocket; avoids uvicorn auto-loading the websockets library
        # (older installs could leave a broken websockets package, causing the server to crash on startup).
        config = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning", ws="none")
        uvicorn.Server(config).run(sockets=[listener])
    except BaseException as e:
        _BACKEND_ERROR = e
        log(f"[Desktop App] Backend stopped due to error: {e}")
    finally:
        listener.close()


def wait_for_server(url: str, timeout: float = 30.0, thread=None) -> bool:
    """Wait for the backend to be ready before opening the desktop window.

    Exit immediately if the backend thread has died, instead of making the user wait for the full `timeout`
    when the model fails to load or the port is already in use.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if thread is not None and not thread.is_alive():
            return False
        if _probe_health(url) is not None:
            return thread is None or thread.is_alive()
        time.sleep(0.3)
    return False


import base64


class DesktopApi:
    """Provides native desktop OS APIs to the frontend via PyWebView."""

    def __init__(self):
        self._window = None
        # Files saved by the app in this session: open_saved only opens these files, so JS
        # cannot open/execute arbitrary paths on the machine.
        self._saved_paths: list[str] = []

    def set_window(self, window):
        self._window = window

    def quit_app(self) -> None:
        """"Exit" button on the terms screen: closes the app if the user does not agree."""
        if self._window is not None:
            self._window.destroy()

    def _ask_save_path(self, filename: str):
        """Native Windows Save As dialog. Returns the path, or None if cancelled."""
        # Determine file format filters
        file_ext = Path(filename).suffix.lower()
        if file_ext == ".mp3":
            file_types = ("Audio MP3 (*.mp3)", "All files (*.*)")
        elif file_ext == ".wav":
            file_types = ("Audio WAV (*.wav)", "All files (*.*)")
        elif file_ext == ".txt":
            file_types = ("Text File (*.txt)", "All files (*.*)")
        elif file_ext == ".zip":
            file_types = ("ZIP (*.zip)", "All files (*.*)")
        else:
            file_types = ("All files (*.*)",)

        # Save directory chosen by the user in Settings (defaults to Downloads).
        from apps import save_settings

        settings = save_settings.load()
        folder = save_settings.effective_folder(settings)
        # "Ask every time" disabled: save directly to the folder without overwriting existing files.
        if not settings["ask"]:
            return str(save_settings.unique_path(folder, filename))
        default_dir = str(folder)

        # Open the native Windows folder/file Save As dialog
        result = self._window.create_file_dialog(
            dialog_type=webview.FileDialog.SAVE,
            directory=default_dir,
            save_filename=filename,
            file_types=file_types,
        )

        # If the user clicks Cancel or closes the dialog
        if not result:
            return None
        return (result if isinstance(result, str) else result[0]) or None

    def save_file(self, filename: str, base64_data: str) -> dict:
        """
        Show the Windows Save As dialog so the user can choose a folder and file name,
        then write binary data directly to disk.
        """
        try:
            if not self._window:
                return {"success": False, "error": "Window is not initialized"}
            save_path = self._ask_save_path(filename)
            if not save_path:
                return {"success": False, "cancelled": True}

            # Strip Data URL prefix if present (data:audio/wav;base64,...)
            if "," in base64_data:
                base64_data = base64_data.split(",", 1)[1]

            file_bytes = base64.b64decode(base64_data)
            with open(save_path, "wb") as f:
                f.write(file_bytes)

            log(f"[Desktop App] File saved successfully: {save_path}")
            return self._after_save(save_path)
        except Exception as e:
            log(f"[Desktop App] Error saving file: {e}")
            return {"success": False, "error": str(e)}

    def save_history_audio(self, entry_id: str, filename: str, fmt: str = "wav") -> dict:
        """Save audio for a history item: Python copies (or converts to MP3) directly from
        the file on disk. A ~2-hour file (~690 MB) sent via base64 like ``save_file`` would exceed
        browser string limits (~537 million characters)."""
        try:
            if not self._window:
                return {"success": False, "error": "Window is not initialized"}
            save_path = self._ask_save_path(filename)
            if not save_path:
                return {"success": False, "cancelled": True}
            from apps import history_store

            out = history_store.export_audio(entry_id, save_path, "mp3" if fmt == "mp3" else "wav")
            log(f"[Desktop App] File saved successfully: {out}")
            return self._after_save(str(out))
        except Exception as e:
            log(f"[Desktop App] Error saving file: {e}")
            return {"success": False, "error": str(getattr(e, "detail", e))}

    # ---- After saving: remember the path, auto-open if enabled by the user ----

    def _after_save(self, path: str) -> dict:
        from apps import save_settings

        self._saved_paths = [p for p in self._saved_paths if p != path][-49:] + [path]
        opened = False
        if save_settings.load()["open_after"]:
            try:
                save_settings.open_path(Path(path))
                opened = True
            except Exception as e:
                log(f"[Desktop App] Could not open the saved file: {e}")
        return {"success": True, "path": path, "name": Path(path).name, "opened": opened}

    def open_saved(self, path: str, reveal: bool = False) -> dict:
        """"Open file" / "Open folder" buttons on the save notification."""
        from apps import save_settings

        if path not in self._saved_paths or not Path(path).exists():
            return {"success": False, "error": "Saved file not found (it may have been deleted or renamed)."}
        try:
            save_settings.open_path(Path(path), reveal=bool(reveal))
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ---- Save file settings (Settings -> Save file) ----

    def get_save_settings(self) -> dict:
        from apps import save_settings

        s = save_settings.load()
        return {**s, "effective_folder": str(save_settings.effective_folder(s)),
                "default_folder": str(save_settings.default_folder())}

    def set_save_settings(self, settings: dict) -> dict:
        from apps import save_settings

        try:
            save_settings.save(settings or {})
            return {"success": True, **self.get_save_settings()}
        except (ValueError, OSError) as e:
            return {"success": False, "error": str(e)}

    def choose_save_folder(self) -> dict:
        """Native OS folder picker dialog; does not save, only returns the path."""
        from apps import save_settings

        if not self._window:
            return {"success": False, "error": "Window is not initialized"}
        result = self._window.create_file_dialog(
            dialog_type=webview.FileDialog.FOLDER,
            directory=str(save_settings.effective_folder()),
        )
        if not result:
            return {"success": False, "cancelled": True}
        return {"success": True, "folder": result if isinstance(result, str) else result[0]}


def _run_in_browser(reason=None) -> None:
    """Fallback: open the UI in the default browser.

    Used when the desktop window cannot be created (missing .NET Framework, WebView2, or
    missing pythonnet runtime files in the package). The backend still serves the full
    React UI at `/`, so the app still works - only the native Save As dialog is unavailable,
    and the frontend falls back to the browser's `showSaveFilePicker`.
    """
    import webbrowser

    if reason is not None:
        log(f"[Desktop App] Could not open desktop window: {reason}")
    log(f"[Desktop App] Falling back to default browser: {BACKEND_URL}")

    opened = False
    try:
        opened = bool(webbrowser.open(BACKEND_URL))
    except Exception as e:
        log(f"[Desktop App] Could not open browser: {e}")

    if not opened:
        # No automatic option left - show the address to the user, do not fail silently.
        _show_error(
            "Hupesi TTS",
            "Could not open the application window or the browser.\n\n"
            f"Please open this address manually in your browser:\n{BACKEND_URL}\n\n"
            "Keep this window/application running while using it.",
        )

    # Backend runs in a daemon thread - the main thread must stay alive for the server to keep running.
    log("[Desktop App] Close this terminal window to stop the application.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        log("[Desktop App] Stopped.")


_RESTART_WAIT_ENV = "CIT_RESTART_WAIT_PID"


def _restart_app() -> None:
    """Launch a new app instance and exit this one (the "Restart" button in Settings).

    The new instance waits for the old one to fully exit (see _wait_for_previous_instance) before binding the port,
    otherwise it sees the old port as busy and moves to the next port.
    """
    import subprocess

    log("[Desktop App] Restart requested from Settings...")
    # PyInstaller bootloader environment variables: if left as-is, the child process thinks it is
    # a subprocess of this instance and reuses its extraction directory.
    env = {k: v for k, v in os.environ.items() if not k.startswith("_PYI_") and k != "_MEIPASS2"}
    env["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
    env[_RESTART_WAIT_ENV] = str(os.getpid())
    env.pop(remote_access.LISTEN_ENV, None)
    if getattr(sys, "frozen", False):
        cmd = [sys.executable, *sys.argv[1:]]
        cwd = str(Path(sys.executable).resolve().parent)
    else:
        cmd = [sys.executable, "-m", "apps.desktop_app", *sys.argv[1:]]
        cwd = str(ROOT_DIR)
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    try:
        subprocess.Popen(cmd, cwd=cwd, env=env, close_fds=True, creationflags=flags)
    except Exception as e:
        log(f"[Desktop App] Could not launch new instance, keeping current instance running: {e}")
        return
    # Exit both the backend thread and the WebView2 window. History/settings are already written
    # to disk atomically, so nothing is lost.
    os._exit(0)


def _wait_for_previous_instance(timeout: float = 20.0) -> None:
    pid_text = os.environ.pop(_RESTART_WAIT_ENV, "")
    if not pid_text.isdigit():
        return
    pid = int(pid_text)
    log(f"[Desktop App] Waiting for previous instance (PID {pid}) to exit...")
    if sys.platform == "win32":
        import ctypes

        SYNCHRONIZE = 0x00100000
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
        if handle:
            kernel32.WaitForSingleObject(handle, int(timeout * 1000))
            kernel32.CloseHandle(handle)
        return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return
        time.sleep(0.2)


def main():
    global PORT, BACKEND_URL, _BACKEND_ERROR

    _wait_for_previous_instance()
    from apps import server_settings
    server_settings.restart_handler = _restart_app

    # A daemon backend belongs to its window; borrowing it breaks on owner exit.
    _BACKEND_ERROR = None
    try:
        listener = _reserve_listener(PORT)
    except Exception as e:
        splash_close()
        _show_error("Hupesi TTS", f"Failed to start: {e}")
        return
    PORT = listener.getsockname()[1]
    BACKEND_URL = f"http://127.0.0.1:{PORT}"
    backend_thread = threading.Thread(target=run_backend, args=(listener,), daemon=True)
    backend_thread.start()

    log("[Desktop App] Starting and checking server...")
    if not wait_for_server(BACKEND_URL, timeout=600.0, thread=backend_thread):
        splash_close()
        reason = str(_BACKEND_ERROR) if _BACKEND_ERROR else "Server did not respond in time."
        _show_error(
            "Hupesi TTS - Failed to start",
            "Could not start the internal server.\n\n"
            f"Details: {reason}\n\n"
            f"See log at: {LOG_FILE}",
        )
        return

    # 4. Linux: open in browser. A dedicated window there requires GTK/WebKit properly installed on
    #    the client machine - fragile with no added benefit. Running in a terminal means closing
    #    the terminal (or Ctrl+C) stops the app.
    #    macOS is different: .app bundles launched from Launchpad have no terminal; opening in a
    #    browser would leave the app running in the background with no way to quit. Mac uses a dedicated window
    #    (pywebview + macOS built-in WebKit, PyObjC bundled) like Windows; if it can't
    #    load, it still falls back to the browser in step 4a.
    if sys.platform.startswith("linux"):
        splash_close()
        log("[Desktop App] UI opened in browser. Keep this terminal window open "
            "while using the app; close it or press Ctrl+C to stop the application.")
        _run_in_browser()
        return

    # 4a. PyWebView could not be loaded (missing .NET Framework / pythonnet) → use browser
    if webview is None:
        splash_close()
        _run_in_browser(_WEBVIEW_IMPORT_ERROR)
        return

    # 4b. PyWebView is available but WebView2 Runtime is missing. Do NOT open a window: pywebview
    #     would silently render it with the IE11 engine, and the React UI would appear blank.
    #     Better to open the browser - fully functional and the user sees something.
    if not webview2_available():
        splash_close()
        _run_in_browser(
            "Microsoft Edge WebView2 Runtime is not installed, so the application window cannot be created. "
            f"Install it from: {WEBVIEW2_DOWNLOAD_URL}"
        )
        return

    # 4c. Log environment info. Without these lines, issues on user machines are
    #     impossible to diagnose - --windowed builds have no console output.
    try:
        import platform
        log(f"[Diagnostics] Windows: {platform.platform()}")
        log(f"[Diagnostics] WebView2 available: {webview2_available()}")
        log(f"[Diagnostics] pywebview: {getattr(webview, '__version__', '?')}")
        from webview.platforms import winforms as _wf
        log(f"[Diagnostics] Actual engine: {getattr(_wf, 'renderer', '?')}"
            f" | is_chromium={getattr(_wf, 'is_chromium', '?')}")
    except Exception as e:
        log(f"[Diagnostics] Could not read engine info: {e}")
    try:
        import mimetypes
        log("[Diagnostics] MIME .js={} .mjs={} .css={}".format(
            mimetypes.guess_type("a.js")[0], mimetypes.guess_type("a.mjs")[0],
            mimetypes.guess_type("a.css")[0]))
    except Exception as e:
        log(f"[Diagnostics] Could not read MIME types: {e}")
    log(f"[Diagnostics] UI served from: {BACKEND_URL}")

    # 5. Create the Desktop API object, native desktop window, and run the UI event loop.
    #    PyWebView's WebView2 backend lives in platforms/winforms.py so it still needs
    #    pythonnet - errors here must not crash the application.
    try:
        log("[Desktop App] Opening Hupesi TTS window...")
        desktop_api = DesktopApi()
        # Kep cua so vao vung lam viec that. Doc duoc thi tu tinh, khong doc duoc thi
        # de pywebview tu xu (CenterScreen) - van chay, chi la co the hoi lech.
        work = _primary_work_area_logical()
        if work:
            geom = fit_window_to_screen(work)
            log(f"[Chan doan] Vung lam viec (logic): {work[2]}x{work[3]} tai ({work[0]},{work[1]})")
            log(f"[Chan doan] Cua so: {geom['width']}x{geom['height']} tai ({geom['x']},{geom['y']})")
        else:
            geom = {"width": WINDOW_PREFERRED[0], "height": WINDOW_PREFERRED[1],
                    "min_size": WINDOW_MIN}
            log("[Chan doan] Khong doc duoc vung lam viec - dung kich thuoc mac dinh.")
        window = webview.create_window(
            title="Hupesi TTS - AI Text to Speech",
            url=BACKEND_URL,
            background_color="#f3f8f8",
            hidden=True,
            resizable=True,
            text_select=True,
            js_api=desktop_api,
            **geom,
        )
        desktop_api.set_window(window)
        revealed = threading.Event()

        def reveal_window():
            if revealed.is_set():
                return
            revealed.set()
            splash_close()
            window.show()

        def reveal_fallback():
            # Do not leave a hidden app behind if WebView2 never signals loaded.
            if not window.events.loaded.wait(15):
                log("[Diagnostics] WebView2 did not report loaded after 15 seconds; showing window for diagnosis.")
                reveal_window()

        window.events.loaded += reveal_window
        threading.Thread(target=reveal_fallback, daemon=True).start()
        # CIT_DEBUG=1 enables devtools to inspect JavaScript errors - the only way to diagnose
        # a "window opens but stays blank" issue on user machines.
        debug = os.environ.get("CIT_DEBUG", "") not in ("", "0")
        log(f"[Diagnostics] Opening window (devtools={debug})...")
        # Windows forces WebView2 (see 4b). macOS: None = default Cocoa/WebKit backend.
        webview.start(gui="edgechromium" if sys.platform == "win32" else None, debug=debug)
        splash_close()
        log("[Diagnostics] Window closed.")
    except Exception as e:
        splash_close()
        _run_in_browser(e)


if __name__ == "__main__":
    main()

