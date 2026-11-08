"""The GPU queue (DESIGN.md §10): the daemon's one way to the decider, which `decider.load_decider` makes."""

from __future__ import annotations

import collections
import threading
import time

from mms import decider
from mms.config import Config
from mms.errors import ApiError, Cancelled

NO_MODEL = "no model is configured: the owner runs `mms model use NAME` (`mms model list` shows them)"


class _Request:
    def __init__(self, state, questions, media):
        self.state, self.questions, self.media = state, questions, media
        self.phase = "waiting"  # waiting, running, dropped (its caller was cancelled, and left)
        self.done = threading.Event()
        self.answers: dict | None = None
        self.error: Exception | None = None


class Gpu:
    """The queue in front of the decider: one worker thread owns it (DESIGN.md §10).

    Requests run in arrival order, so jobs that each keep one waiting take turns window by window. The worker loads
    the model at the first request and unloads it after `idle_s` with none. A load that fails fails the waiting
    requests with `decider`, and the next request tries again. A decider whose `decide` fails with `decider` is let go
    (ended outside the lock) and the next request loads a new one, so a restarted worker is always checked by `load`.
    """

    def __init__(self, config: Config):
        self.home = config.home
        self.model = config.decider_model
        self.url, self.envs = config.decider_url, config.decider_envs
        self.idle_s = config.decider_idle_s
        self._cv = threading.Condition()
        self._queue: collections.deque[_Request] = collections.deque()
        self._decider = None
        self._about: tuple[tuple, decider.About] | None = None  # the (model, url, envs) it was read for, and its facts
        self._loading = False
        self._error: str | None = None
        self._last = time.monotonic()
        self._reload = False
        self._stopping = False
        self._worker: threading.Thread | None = None

    def apply(self, config: Config) -> None:
        """A new `[decider] model`, `url` or `envs` unloads the old one; `idle_s` counts from the next idle."""
        with self._cv:
            if (config.decider_model, config.decider_url, config.decider_envs) != (self.model, self.url, self.envs):
                self._reload = True
                self._error = None  # an error belongs to the model that raised it
            self.model, self.url, self.envs = config.decider_model, config.decider_url, config.decider_envs
            self.idle_s = config.decider_idle_s
            self._cv.notify_all()

    def status(self) -> dict:
        with self._cv:
            if self._loading:
                state = "loading"
            elif self._decider is not None:
                state = "ready"
            else:
                state = "failed" if self._error else "unloaded"
            return {"model": self.model, "state": state, "error": self._error, "waiting": len(self._queue)}

    def about(self) -> decider.About:
        """The facts of the configured model, read without loading it: a decider is constructed only to read its
        `about`, and the result is kept for this model, url and envs. A model that cannot be made fails with `decider`."""
        with self._cv:
            model, url, envs = self.model, self.url, self.envs
            if not model:
                raise ApiError("decider", NO_MODEL, hint="ask the owner")
            if self._about is not None and self._about[0] == (model, url, envs):
                return self._about[1]
        try:
            about = decider.load_decider(model, self.home, url=url, envs=envs).about
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            with self._cv:
                self._error = message
            raise ApiError("decider", f"the decider could not load: {message}") from None
        with self._cv:
            self._about = ((model, url, envs), about)
            if self._decider is None and not self._loading and (self.model, self.url, self.envs) == (model, url, envs):
                self._error = None  # the model can be made now: no earlier failure of it stands
        return about

    def check(self, questions: dict) -> None:
        """`check_of` for the configured model: `ValueError` naming what it cannot take. Nothing is loaded."""
        with self._cv:
            model = self.model
        decider.check_of(model, questions)

    def decide(self, state, questions: dict, media: list, cancelled=lambda: False) -> dict:
        """Queue one window and wait for its answers. A cancelled job does not wait: its request is dropped from the queue,
        or if the worker has it (a model may be loading, 20 s) the worker finishes it for nobody, and `Cancelled` is raised
        at once."""
        request = _Request(state, questions, media)
        with self._cv:
            if self._stopping:
                raise ApiError("decider", "the daemon is stopping")
            if not self.model:
                raise ApiError("decider", NO_MODEL, hint="ask the owner")
            self._queue.append(request)
            if self._worker is None:
                self._worker = threading.Thread(target=self._work, name="gpu", daemon=True)
                self._worker.start()
            self._cv.notify_all()
        while not request.done.wait(0.05):
            if cancelled():
                with self._cv:
                    if request.phase == "waiting":
                        self._queue.remove(request)
                    request.phase = "dropped"
                raise Cancelled()
        if cancelled():
            raise Cancelled()
        if request.error is not None:
            raise request.error
        return request.answers

    def stop(self) -> None:
        with self._cv:
            self._stopping = True
            for request in self._queue:
                request.error = ApiError("decider", "the daemon is stopping")
                request.done.set()
            self._queue.clear()
            self._cv.notify_all()
        if self._worker is not None:
            self._worker.join(30)
        with self._cv:
            gone = self._decider
            self._decider = None
        self._unload(gone)

    # the worker

    @staticmethod
    def _unload(gone) -> None:
        """End a decider that `Gpu` has let go of. Never under the lock: a worker takes seconds to end."""
        if gone is not None and hasattr(gone, "unload"):
            gone.unload()

    def _work(self) -> None:
        while True:
            gone = request = None
            with self._cv:
                while True:
                    if self._stopping:
                        return
                    if self._reload:  # between windows: a model that changed is loaded for the next one
                        self._reload = False
                        gone, self._decider = self._decider, None
                        break
                    if self._queue:
                        request = self._queue.popleft()
                        request.phase = "running"
                        break
                    if self._decider is None:
                        self._cv.wait()
                        continue
                    left = self._last + self.idle_s - time.monotonic()
                    if left <= 0:
                        gone, self._decider = self._decider, None
                        break
                    self._cv.wait(left)
            self._unload(gone)
            if request is not None:
                self._run(request)

    def _run(self, request: _Request) -> None:
        failed = None  # a decider whose `decide` raised ApiError is let go, so the next request makes and loads a new one
        try:
            if self._decider is None:
                self._load()
            current = self._decider
            try:
                request.answers = current.decide(request.state, request.questions, request.media)
            except ApiError:
                failed = current
                raise
        except (ValueError, TypeError) as error:  # the questions do not fit this window
            request.error = error
        except ApiError as error:
            request.error = error
        except Exception as error:
            request.error = ApiError("decider", f"{type(error).__name__}: {error}")
        with self._cv:
            if failed is not None and self._decider is failed:
                self._decider = None
            self._last = time.monotonic()
            request.done.set()
        self._unload(failed)

    def _load(self) -> None:
        with self._cv:
            self._loading, model, url, envs = True, self.model, self.url, self.envs
        try:
            made = decider.load_decider(model, self.home, url=url, envs=envs)
            if hasattr(made, "load"):
                made.load()
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            with self._cv:
                self._loading, self._error = False, message
                for waiting in self._queue:  # a load that fails fails the waiting requests with it
                    waiting.error = ApiError("decider", f"the decider could not load: {message}")
                    waiting.done.set()
                self._queue.clear()
            raise ApiError("decider", f"the decider could not load: {message}") from None
        with self._cv:
            self._decider, self._loading, self._error = made, False, None
