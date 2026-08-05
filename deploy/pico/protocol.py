"""Wire format for the PICO 4 Ultra SMPL pose stream.

Mirrors ``gear_sonic.utils.teleop.zmq.zmq_planner_sender.pack_pose_message``
so that ``gamc`` can consume the stream without importing ``gear_sonic``.

Layout of a single ZMQ PUB message::

    [topic bytes]  e.g. b"pose"
    [HEADER_SIZE-byte JSON header, right-padded with b"\\x00"]
    [concatenated little-endian binary field payloads]

The JSON header describes every field in payload order::

    {"v": 3, "endian": "le", "count": 1,
     "fields": [{"name": "joint_pos", "dtype": "f32", "shape": [5, 29]}, ...]}
"""

from __future__ import annotations

import json

import numpy as np

# NOTE: the sender's docstring says 1024, but the module constant is 1280.
HEADER_SIZE = 1280

DEFAULT_TOPIC = "pose"
DEFAULT_PORT = 5555

_DTYPE_MAP: dict[str, np.dtype] = {
    "f32": np.dtype("<f4"),
    "f64": np.dtype("<f8"),
    "i32": np.dtype("<i4"),
    "i64": np.dtype("<i8"),
    "u8": np.dtype("u1"),
    "bool": np.dtype("bool"),
}


class PicoProtocolError(RuntimeError):
    """Raised when a received frame cannot be decoded."""


def unpack_pose_message(data: bytes, topic: str = DEFAULT_TOPIC) -> dict[str, np.ndarray]:
    """Decode one packed pose message into ``{field_name: ndarray}``.

    ``data`` may still carry the leading topic prefix; it is stripped when
    present so this works with both raw ``socket.recv()`` output and the
    already-stripped output of a poller.
    """
    topic_bytes = topic.encode("utf-8")
    if data.startswith(topic_bytes):
        data = data[len(topic_bytes):]

    if len(data) < HEADER_SIZE:
        raise PicoProtocolError(
            f"message too short: {len(data)} bytes < {HEADER_SIZE}-byte header"
        )

    header_raw = data[:HEADER_SIZE].rstrip(b"\x00")
    try:
        header = json.loads(header_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PicoProtocolError(f"could not parse JSON header: {exc}") from exc

    if header.get("endian", "le") != "le":
        raise PicoProtocolError("only little-endian payloads are supported")

    payload = data[HEADER_SIZE:]
    out: dict[str, np.ndarray] = {}
    offset = 0

    for field in header.get("fields", []):
        name = field["name"]
        dtype_str = field["dtype"]
        shape = tuple(int(s) for s in field["shape"])

        dtype = _DTYPE_MAP.get(dtype_str)
        if dtype is None:
            raise PicoProtocolError(f"unsupported dtype {dtype_str!r} for field {name!r}")

        count = int(np.prod(shape)) if shape else 1
        nbytes = count * dtype.itemsize
        if offset + nbytes > len(payload):
            raise PicoProtocolError(
                f"payload truncated at field {name!r}: "
                f"need {nbytes} bytes at offset {offset}, have {len(payload) - offset}"
            )

        arr = np.frombuffer(payload, dtype=dtype, count=count, offset=offset)
        out[name] = arr.reshape(shape).copy()
        offset += nbytes

    return out


def pack_pose_message(
    pose_data: dict[str, np.ndarray],
    topic: str = DEFAULT_TOPIC,
    version: int = 3,
) -> bytes:
    """Inverse of :func:`unpack_pose_message`.

    Only needed by :mod:`deploy.pico.mock_server`; the real stream is packed
    on the PICO host by ``gear_sonic``.
    """
    reverse_map = {
        np.dtype("float32"): "f32",
        np.dtype("float64"): "f64",
        np.dtype("int32"): "i32",
        np.dtype("int64"): "i64",
        np.dtype("uint8"): "u8",
        np.dtype("bool"): "bool",
    }

    fields = []
    blobs = []
    for name, value in pose_data.items():
        value = np.ascontiguousarray(value)
        dtype_str = reverse_map.get(value.dtype)
        if dtype_str is None:
            value = value.astype(np.float32)
            dtype_str = "f32"
        fields.append({"name": name, "dtype": dtype_str, "shape": list(value.shape)})
        blobs.append(value.tobytes())

    header = {"v": version, "endian": "le", "count": 1, "fields": fields}
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(header_json) > HEADER_SIZE:
        raise PicoProtocolError(f"header too large: {len(header_json)} > {HEADER_SIZE}")

    return topic.encode("utf-8") + header_json.ljust(HEADER_SIZE, b"\x00") + b"".join(blobs)
