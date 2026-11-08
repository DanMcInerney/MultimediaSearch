"""What the daemon's parts and its callers raise: the API's one error (DESIGN.md §11), which the daemon raises and the
client raises again from the reply, and a cancelled job. No imports, so the adapter can load it."""

from __future__ import annotations

STATUS = {"bad_request": 400, "unauthorized": 401, "not_found": 404, "wrong_kind": 409, "would_delete": 409,
          "config_error": 409, "too_big": 413, "floor": 507, "decider": 503, "internal": 500}


class ApiError(Exception):
    """An error of the API: `code` is one of `STATUS`, `field` names the parameter at fault, and `hint` says what the
    caller can do. The client's own conditions (client.py) carry codes that are never sent, so they have no status."""

    def __init__(self, code: str, message: str, field: str | None = None, hint: str | None = None):
        super().__init__(message)
        self.code, self.message, self.field, self.hint = code, message, field, hint

    @property
    def status(self) -> int | None:
        return STATUS.get(self.code)

    def body(self) -> dict:
        error = {"code": self.code, "message": self.message}
        if self.field is not None:
            error["field"] = self.field
        if self.hint is not None:
            error["hint"] = self.hint
        return error


class Cancelled(Exception):
    """The job was cancelled; raised inside it at its next call to the decider."""
