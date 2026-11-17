from __future__ import annotations

from typing import Callable, Iterable, Tuple

StartResponse = Callable[[str, Iterable[Tuple[str, str]], object], Callable[..., object]]

class PrefixMiddleware:
    """WSGI middleware to support deployment under a subpath via X-Forwarded-Prefix.

    If the reverse proxy sets X-Forwarded-Prefix: /anispeakarchive, this middleware will:
      - set SCRIPT_NAME to that prefix
      - strip the prefix from PATH_INFO (so Flask routing matches)
    """

    def __init__(self, app: Callable):
        self.app = app

    def __call__(self, environ: dict, start_response: StartResponse):
        prefix = str(environ.get("HTTP_X_FORWARDED_PREFIX", "") or "").strip()
        if prefix:
            assert prefix.startswith("/"), "X-Forwarded-Prefix must start with '/'"
            # normalize: '/foo/' -> '/foo'
            prefix = prefix.rstrip("/")
            assert prefix == "" or prefix.startswith("/"), "normalized prefix must be empty or start with '/'"
            environ["SCRIPT_NAME"] = prefix

            path_info = str(environ.get("PATH_INFO", "") or "")
            # Some proxies forward PATH_INFO already stripped; handle both.
            if path_info.startswith(prefix):
                new_path = path_info[len(prefix):]
                environ["PATH_INFO"] = new_path if new_path.startswith("/") else ("/" + new_path if new_path else "/")
        return self.app(environ, start_response)
