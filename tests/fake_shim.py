"""A stand-in for H2O-Lightning-4B's server (TESTBED.md, "No GPU"; DESIGN.md §13.3): a stdlib HTTP server on 127.0.0.1 and a
free port that speaks what `h2o_lightning_shim.py` does at `/health` and `/v1/systemone`, in the card's answer shapes, with
no model behind it.

- `/health` answers 503 (`warming`) for the first `warming` asks, then 200 with `model`.
- `/v1/systemone` takes `{state, questions, images}`: a body without `state` or `questions`, more than 4 images, or an
  image that is not a PNG data URI is a 422 with a `detail`. Each image is decoded with PIL, and the request is recorded in
  `requests`: the state, the questions, and per image its size and the sha256 of its pixels (`tobytes()`), which equals
  what was sent when the PNG is lossless.
- An answer is yes for a question whose instructions say `red`, when a picture is mostly red (otherwise no): a yes/no
  0.95 or 0.05, a choice its first option or its last, a score its top level or 0, a box found or not.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

MODEL_ID = "h2oai/h2o-lightning-4b"
MAX_IMAGES = 4
PNG_URI = "data:image/png;base64,"
BOX = {"left": 0.1, "top": 0.2, "right": 0.6, "bottom": 0.7}
FULL = {"left": 0.0, "top": 0.0, "right": 1.0, "bottom": 1.0}


def red(image: Image.Image) -> bool:
    r, g, b = np.asarray(image, dtype=float).reshape(-1, 3).mean(axis=0)
    return r > 120 and r > 2 * g and r > 2 * b


def answer(question: dict, yes: bool) -> dict:
    """The card's answer to one question, in the shim's shapes."""
    kind = question["type"]
    if kind == "noul":
        return {"type": "noul", "noul": 0.95 if yes else 0.05}
    if kind == "choice":
        names = list(question["criteria"])
        pick = names[0] if yes else names[-1]
        odds = {name: (0.8 if name == pick else 0.2 / (len(names) - 1)) for name in names} if len(names) > 1 else {pick: 1.0}
        return {"type": "choice", "choice": pick, "confidence": 0.6, "probabilities": odds}
    if kind == "score":
        levels = len(question["criteria"])
        top = levels - 1 if yes else 0
        odds = {str(i): (0.9 if i == top else 0.1 / (levels - 1)) for i in range(levels)}
        return {"type": "score", "score": sum(i * p for i, p in enumerate(odds.values())),
                "legend": {str(i): text for i, text in enumerate(question["criteria"])}, "probabilities": odds,
                "confidence": 0.9}
    if kind == "box":
        if yes:
            return {"type": "box", "box": BOX, "confidence": 0.8, "found": True, "present": 0.9}
        return {"type": "box", "box": FULL, "confidence": 0.0, "found": False, "present": 0.05}
    raise ValueError(f"the fake shim answers noul, choice, score and box, not {kind!r}")


class FakeShim:
    """The server. `with FakeShim() as shim:` runs it and stops it; `shim.url` is where it is."""

    def __init__(self, *, warming: int = 0, model: str = MODEL_ID):
        self.warming, self.model = warming, model
        self.health_asks = 0
        self.requests: list[dict] = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.thread = threading.Thread(target=self.server.serve_forever, name="fake-shim", daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def __enter__(self) -> FakeShim:
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(10)

    def decide(self, body) -> dict:
        """The reply to a `/v1/systemone` body, or a ValueError (a 422) for one that does not fit."""
        if not isinstance(body, dict) or "state" not in body or not isinstance(body.get("questions"), dict):
            raise ValueError("expected a JSON object with `state` and a `questions` object")
        uris = body.get("images") or []
        if len(uris) > MAX_IMAGES:
            raise ValueError(f"{len(uris)} images; this model reads at most {MAX_IMAGES}")
        images, seen = [], []
        for uri in uris:
            if not isinstance(uri, str) or not uri.startswith(PNG_URI):
                raise ValueError("an image is a data:image/png;base64 URI")
            with Image.open(io.BytesIO(base64.b64decode(uri[len(PNG_URI):], validate=True))) as opened:
                images.append(opened.convert("RGB"))
            seen.append({"size": list(images[-1].size), "sha256": hashlib.sha256(images[-1].tobytes()).hexdigest()})
        self.requests.append({"state": body["state"], "questions": body["questions"], "images": seen})
        yes = any(red(image) for image in images)
        return {"model": self.model,
                "answers": {name: answer(question, yes and "red" in str(question.get("instructions")).lower())
                            for name, question in body["questions"].items()}}

    def _handler(self):
        shim = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *args):
                pass

            def send(self, code: int, body: dict) -> None:
                data = json.dumps(body).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if not self.path.startswith("/health"):
                    return self.send(404, {"error": "not found"})
                shim.health_asks += 1
                if shim.health_asks <= shim.warming:
                    return self.send(503, {"ok": False, "model": shim.model, "warming": True})
                self.send(200, {"ok": True, "model": shim.model})

            def do_POST(self):
                if not self.path.startswith("/v1/systemone"):
                    return self.send(404, {"error": "not found"})
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                try:
                    self.send(200, shim.decide(json.loads(raw or b"null")))
                except ValueError as error:
                    self.send(422, {"detail": str(error)})

        return Handler
