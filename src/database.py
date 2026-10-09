"""Lazy, serialized access to a database connection."""

import threading
import time


class DatabaseUnavailable(Exception):
    """The database could not be initialized; a later attempt may succeed."""


class ConnectionManager:
    def __init__(self, factory, cooldown=15, clock=time.monotonic, recovery=None):
        self._factory = factory
        self._cooldown = cooldown
        self._clock = clock
        self._lock = threading.Lock()
        self._connection = None
        self._retry_at = 0
        self._recovery = recovery

    def get(self):
        with self._lock:
            if self._connection is not None:
                return self._connection
            if self._clock() < self._retry_at:
                raise DatabaseUnavailable("Database retry cooldown is active")
            try:
                self._connection = self._factory()
            except Exception as exc:
                self._retry_at = self._clock() + self._cooldown
                if self._recovery is not None:
                    self._recovery(exc)
                raise DatabaseUnavailable("Database initialization failed") from exc
            return self._connection

    def invalidate(self, connection):
        with self._lock:
            # An older session must not invalidate a replacement connection.
            if self._connection is connection:
                self._connection = None
                self._retry_at = self._clock() + self._cooldown
                try:
                    connection.close()
                except Exception:
                    pass
