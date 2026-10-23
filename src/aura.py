"""Optional Aura management API recovery; no network calls at import time."""

import logging
import os
import threading
import time
from urllib.parse import quote

import requests


logger = logging.getLogger(__name__)


class AuraResumer:
    BASE_URL = "https://api.neo4j.io"

    def __init__(self, instance_id, client_id, client_secret, *, enabled=False,
                 http=requests, clock=time.monotonic, cooldown=60):
        self.enabled = enabled
        self._instance_id = instance_id
        self._client_id = client_id
        self._client_secret = client_secret
        self._http = http
        self._clock = clock
        self._cooldown = cooldown
        self._lock = threading.Lock()
        self._retry_at = 0
        self._token = None
        self._token_expires_at = 0
        self.status = None

    @classmethod
    def from_env(cls):
        return cls(
            os.getenv("AURA_INSTANCEID"), os.getenv("AURA_CLIENT_ID"),
            os.getenv("AURA_CLIENT_SECRET"),
            enabled=os.getenv("AURA_AUTO_RESUME", "false").lower() == "true",
        )

    def _headers(self):
        if self._token is None or self._clock() >= self._token_expires_at:
            response = self._http.post(
                f"{self.BASE_URL}/oauth/token",
                auth=(self._client_id, self._client_secret),
                data={"grant_type": "client_credentials"}, timeout=(5, 15),
            )
            response.raise_for_status()
            data = response.json()
            token = data["access_token"]
            expires_in = float(data["expires_in"])
            self._token = token
            self._token_expires_at = self._clock() + max(0, expires_in - 60)
        return {"Authorization": f"Bearer {self._token}"}

    def resume_if_paused(self):
        """Check once per cooldown, resume only PAUSED, and contain API errors."""
        if not self.enabled:
            return
        with self._lock:
            if self._clock() < self._retry_at:
                return
            self._retry_at = self._clock() + self._cooldown
            try:
                if not all((self._instance_id, self._client_id, self._client_secret)):
                    raise ValueError("Aura auto-resume credentials are incomplete")
                headers = self._headers()
                url = f"{self.BASE_URL}/v1/instances/{quote(self._instance_id, safe='')}"
                response = self._http.get(url, headers=headers, timeout=(5, 15))
                response.raise_for_status()
                self.status = response.json()["data"]["status"].lower()
                if self.status == "paused":
                    response = self._http.post(
                        f"{url}/resume", headers=headers, timeout=(5, 15)
                    )
                    response.raise_for_status()
                    self.status = "resuming"
            except Exception as exc:
                self.status = None
                if isinstance(exc, requests.HTTPError) and exc.response is not None:
                    if exc.response.status_code in (401, 403):
                        self._token = None
                # Never log response bodies, tokens, or credential-bearing errors.
                logger.warning("Aura recovery failed (%s); will retry later", type(exc).__name__)

    @property
    def starting(self):
        return self.status in ("resuming", "updating", "loading")
