"""
offline_mode.py — disable sockets/HTTP to keep the process hermetic.
Import this at the very top of entrypoints (train, repl, export).
"""
import builtins

# Block network: monkeypatch socket to raise immediately.
def _block(*args, **kwargs):
    raise RuntimeError("Networking is disabled in offline mode.")

try:
    import socket
    socket.socket = _block  # type: ignore
    socket.create_connection = _block  # type: ignore
except Exception:
    pass

# Optional: block urllib/requests if present (won't import them here).
