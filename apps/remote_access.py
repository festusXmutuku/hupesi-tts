"""Allow other machines (LAN / Internet) to access the API/UI, with an API key. Disabled by default.

Rules:
- Requests from this machine (loopback, no proxy forwarding headers) are unaffected.
- Requests from other machines, or with proxy/tunnel forwarding headers, are allowed
  when remote access is enabled (full UI and API access). When disabled, they get 403.
- Disabling takes effect immediately. Enabling may require a restart so the server
  listens on 0.0.0.0 (all interfaces) instead of 127.0.0.1 (local only).
"""
from __future__ import annotations

import hmac
import ipaddress
import json
import os
import secrets
import socket
import tempfile
import threading
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

# Matches the public API list in hupesi-tts.md / tests/test_public_api_docs.py.
PUBLIC_PATHS = frozenset({
    "/health",
    "/voices",
    "/stream",
    "/api/tts/generate",
    "/api/tts/generate-with-subtitless",
    "/api/tts/conversation",
})
# Documentation describes the public API and can be read without a key (Swagger fetches openapi.json itself).
DOC_PATHS = frozenset({"/hupesi-tts.md", "/docs", "/docs/oauth2-redirect", "/openapi.json"})

_FORWARD_HEADERS = (b"x-forwarded-for", b"forwarded", b"x-real-ip", b"cf-connecting-ip", b"x-forwarded-host")
_LOCAL_HOSTNAMES = {"127.0.0.1", "localhost", "::1"}
LISTEN_ENV = "CIT_LISTEN_HOST"

_lock = threading.Lock()
_state: dict | None = None


def config_path() -> Path:
    return Path(os.getenv("LOCALAPPDATA") or Path.home()) / "HupesiTTS" / "remote-access.json"


def new_key() -> str:
    return "cit_" + secrets.token_urlsafe(24)


def _write(state: dict) -> None:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
            json.dump(state, file)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def current() -> dict:
    """{"enabled": bool, "api_key": str}. Generates and saves a key on first use."""
    global _state
    with _lock:
        if _state is None:
            state = {"enabled": False, "api_key": ""}
            try:
                data = json.loads(config_path().read_text(encoding="utf-8"))
                state["enabled"] = data.get("enabled") is True
                if isinstance(data.get("api_key"), str) and len(data["api_key"]) >= 20:
                    state["api_key"] = data["api_key"]
            except (OSError, ValueError, AttributeError):
                pass
            if not state["api_key"]:
                state["api_key"] = new_key()
                try:
                    _write(state)
                except OSError:
                    pass  # Cannot write: use the in-memory key for this session.
            _state = state
        return dict(_state)


def _update(**changes) -> dict:
    global _state
    state = {**current(), **changes}
    _write(state)  # If writing fails, keep the previous state.
    with _lock:
        _state = state
    return dict(state)


def reset_cache() -> None:
    global _state
    with _lock:
        _state = None


def listen_host() -> str:
    """Address the server should listen on at startup."""
    return "0.0.0.0" if current()["enabled"] else "127.0.0.1"


def mark_listening(host: str) -> None:
    os.environ[LISTEN_ENV] = host


def is_listening_all() -> bool:
    return os.environ.get(LISTEN_ENV) == "0.0.0.0"


def primary_address() -> str | None:
    """IPv4 address of the primary network interface (the one the OS uses for outbound traffic).
    Other IPs are often virtual adapters (VirtualBox 192.168.56.1, Hyper-V, VPN) that other machines cannot reach."""
    try:
        # No packets are sent: the UDP connect only lets the OS choose the interface via the routing table.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
    except OSError:
        return None
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return None
    return None if addr.is_loopback or addr.is_unspecified or addr.is_link_local else ip


def lan_addresses() -> list[str]:
    """LAN IPv4 addresses of this machine, with the primary interface first."""
    found: list[str] = []
    primary = primary_address()
    if primary:
        found.append(primary)
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.append(info[4][0])
    except OSError:
        pass
    result = []
    for ip in found:
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            continue
        if not (addr.is_loopback or addr.is_link_local or addr.is_unspecified) and ip not in result:
            result.append(ip)
    return result


def _is_loopback(host: str | None) -> bool:
    if host == "testclient":  # Starlette TestClient fake client; real sockets always use an IP.
        return True
    try:
        addr = ipaddress.ip_address(host or "")
    except ValueError:
        return False
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
        addr = addr.ipv4_mapped
    return addr.is_loopback


def is_remote(scope) -> bool:
    client = scope.get("client")
    if not _is_loopback(client[0] if client else None):
        return True
    return any(name in _FORWARD_HEADERS for name, _ in scope.get("headers", []))


def _key_from(scope) -> str:
    headers = {name: value.decode("latin-1") for name, value in scope.get("headers", [])}
    key = headers.get(b"x-api-key", "").strip()
    if key:
        return key
    auth = headers.get(b"authorization", "")
    if auth[:7].lower() == "bearer ":
        return auth[7:].strip()
    # For <audio src="/stream?...&api_key=..."> tags - these cannot send headers.
    query = parse_qs(scope.get("query_string", b"").decode("latin-1"))
    return (query.get("api_key") or [""])[0].strip()


class RemoteAccessMiddleware:
    """Control access from other machines. Must be the outermost middleware."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket") or not is_remote(scope):
            await self.app(scope, receive, send)
            return
        state = current()
        if not state["enabled"]:
            await _deny(
                scope,
                send,
                403,
                "Remote access is disabled. Enable it in Hupesi TTS settings.",
            )
            return
        # Remote access enabled: allow the full UI and API to be used from other machines.
        await self.app(scope, receive, send)


async def _deny(scope, send, status: int, detail: str) -> None:
    if scope["type"] == "websocket":
        await send({"type": "websocket.close", "code": 1008})
        return
    body = json.dumps({"detail": detail}, ensure_ascii=False).encode("utf-8")
    headers = [
        (b"content-type", b"application/json; charset=utf-8"),
        (b"content-length", str(len(body)).encode()),
        # Error returned before CORSMiddleware: add manually so the browser can read the message.
        (b"access-control-allow-origin", b"*"),
    ]
    if status == 401:
        headers.append((b"www-authenticate", b'Bearer realm="Hupesi TTS"'))
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


# ---- Settings API (UI on this machine only) ----

router = APIRouter()


def require_this_machine(request: Request) -> None:
    """Restrict sensitive settings changes to trusted UI access.

    When remote access is disabled, only the local app/UI may change settings.
    When remote access is explicitly enabled, allow remote UI access as requested.
    """
    state = current()
    if state.get("enabled"):
        return
    host = (request.headers.get("host") or "").rsplit(":", 1)[0].strip("[]").lower()
    if host not in _LOCAL_HOSTNAMES:
        raise HTTPException(status_code=403, detail="This can only be changed from the app on this machine.")
    origin = request.headers.get("origin")
    if origin and origin != "null":
        parts = urlsplit(origin)
        if parts.scheme not in ("http", "https") or (parts.hostname or "").lower() not in _LOCAL_HOSTNAMES:
            raise HTTPException(status_code=403, detail="This can only be changed from the app on this machine.")


def _status(request: Request) -> dict:
    state = current()
    return {
        "enabled": state["enabled"],
        "api_key": state["api_key"],
        "listening_all": is_listening_all(),
        "port": request.url.port or 80,
        "addresses": lan_addresses(),
        "primary_address": primary_address(),
    }


class RemoteAccessUpdate(BaseModel):
    enabled: bool


@router.get("/api/remote-access")
def get_remote_access(request: Request):
    require_this_machine(request)
    return _status(request)


@router.put("/api/remote-access")
def set_remote_access(update: RemoteAccessUpdate, request: Request):
    require_this_machine(request)
    try:
        _update(enabled=update.enabled)
    except OSError as error:
        raise HTTPException(status_code=500, detail="Could not save settings. Check configuration write permissions.") from error
    return _status(request)


@router.post("/api/remote-access/regenerate-key")
def regenerate_key(request: Request):
    require_this_machine(request)
    try:
        _update(api_key=new_key())
    except OSError as error:
        raise HTTPException(status_code=500, detail="Could not save the new key.") from error
    return _status(request)
