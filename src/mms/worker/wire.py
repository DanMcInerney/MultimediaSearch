"""The frames between the daemon and a worker (DESIGN.md §13.3). Standard library at the top; numpy and PIL are imported
inside the media functions, so an env without them can still read frames.

    frame  = b"MSW1" | header length (u32, little-endian) | header (UTF-8 JSON) | blob 0 | blob 1 | ...
    header = {"op": ..., ..., "blobs": [len0, len1, ...]}      # "blobs" is left out when there are none
"""

from __future__ import annotations

import json
import struct

MAGIC = b"MSW1"
MAX_HEADER = 16 << 20  # bytes of header
MAX_BLOBS = 256 << 20  # bytes of blobs in a frame (a file window is at most 64 MB)


class WireError(Exception):
    """The bytes are not a frame, or a frame is over the limits. `found` holds the first bytes when they are not a frame."""

    def __init__(self, message: str, found: bytes | None = None):
        super().__init__(message)
        self.found = found


def write_frame(stream, header: dict, blobs=()) -> None:
    """Write one frame to a binary stream and flush it."""
    blobs = [memoryview(blob).cast("B") for blob in blobs]
    header = dict(header)
    if blobs:
        header["blobs"] = [len(blob) for blob in blobs]
    text = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(text) > MAX_HEADER:
        raise WireError(f"a header of {len(text)} bytes is over the limit of {MAX_HEADER}")
    if sum(len(blob) for blob in blobs) > MAX_BLOBS:
        raise WireError(f"blobs of {sum(len(blob) for blob in blobs)} bytes are over the limit of {MAX_BLOBS}")
    stream.write(MAGIC + struct.pack("<I", len(text)) + text)
    for blob in blobs:
        stream.write(blob)
    stream.flush()


def _read(stream, size: int) -> bytes:
    data = stream.read(size)
    if len(data) < size:
        raise WireError(f"the stream ended {size - len(data)} bytes short inside a frame")
    return data


def read_frame(stream) -> tuple[dict, list[bytes]] | None:
    """Read one frame from a binary buffered stream: its header and blobs, or None when the stream ends between
    frames. Raises `WireError` for bytes that are not a frame (the message holds the first 16), a header or blobs over
    the limits, and a stream that ends inside a frame."""
    head = stream.peek(1)[:16]  # what has arrived, so a stray short write is named rather than waited for
    if not head:
        return None
    if head[:4] != MAGIC[:len(head[:4])]:
        raise WireError(f"not a frame: {head!r}", found=head)
    prefix = _read(stream, 8)
    (size,) = struct.unpack("<I", prefix[4:])
    if size > MAX_HEADER:
        raise WireError(f"a header of {size} bytes is over the limit of {MAX_HEADER}")
    try:
        header = json.loads(_read(stream, size).decode("utf-8"))
    except ValueError as error:
        raise WireError(f"the header is not JSON: {error}") from None
    sizes = header.get("blobs", []) if isinstance(header, dict) else None
    if (not isinstance(sizes, list)
            or not all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in sizes)):
        raise WireError("the header is not an object with a list of blob sizes")
    if sum(sizes) > MAX_BLOBS:
        raise WireError(f"blobs of {sum(sizes)} bytes are over the limit of {MAX_BLOBS}")
    return header, [_read(stream, n) for n in sizes]


def encode_media(media: list) -> tuple[list[dict], list]:
    """A window's media (a PIL image, `{"type": "audio", ...}` or `{"type": "file", ...}`) as descriptors and the
    blobs they point to. Raises `ValueError` for a medium mms does not send."""
    import numpy as np
    from PIL import Image

    descriptors, blobs = [], []
    for medium in media:
        if isinstance(medium, Image.Image):
            if medium.mode != "RGB":
                raise ValueError(f"a picture is sent as RGB, not {medium.mode}")
            descriptors.append({"kind": "image", "mode": "RGB", "size": list(medium.size), "blob": len(blobs)})
            blobs.append(medium.tobytes())
        elif isinstance(medium, dict) and medium.get("type") == "audio":
            samples = np.ascontiguousarray(medium["audio"], dtype="<f4")
            descriptors.append({"kind": "audio", "rate": int(medium["sampling_rate"]), "samples": int(samples.size),
                                "blob": len(blobs)})
            blobs.append(samples.tobytes())
        elif isinstance(medium, dict) and medium.get("type") == "file":
            descriptors.append({"kind": "file", "media": medium.get("media"), "name": medium["name"],
                                "blob": len(blobs)})
            blobs.append(bytes(medium["data"]))
        else:
            raise ValueError(f"mms sends pictures, sound and files, not {type(medium).__name__}")
    return descriptors, blobs


def decode_media(descriptors: list, blobs: list) -> list:
    """The inverse of `encode_media`: pictures as PIL images, sound and files as the dicts the deciders take. Raises
    `ValueError` for a descriptor that does not fit its blob."""
    import numpy as np
    from PIL import Image

    media = []
    for descriptor in descriptors:
        kind, blob = descriptor.get("kind"), blobs[descriptor["blob"]]
        if kind == "image":
            width, height = descriptor["size"]
            if descriptor.get("mode") != "RGB" or len(blob) != width * height * 3:
                raise ValueError(f"a picture of {width}x{height} {descriptor.get('mode')} is not {len(blob)} bytes")
            media.append(Image.frombytes("RGB", (width, height), blob))
        elif kind == "audio":
            if len(blob) != descriptor["samples"] * 4:
                raise ValueError(f"sound of {descriptor['samples']} samples is not {len(blob)} bytes")
            media.append({"type": "audio", "audio": np.frombuffer(blob, "<f4").copy(),
                          "sampling_rate": descriptor["rate"]})
        elif kind == "file":
            media.append({"type": "file", "media": descriptor.get("media"), "name": descriptor["name"], "data": blob})
        else:
            raise ValueError(f"a medium of kind {kind!r}")
    return media
