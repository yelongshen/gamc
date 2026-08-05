"""ZMQ subscriber for the PICO 4 Ultra SMPL pose stream.

The publisher is ``gear_sonic/scripts/pico_manager_thread_server.py`` run with
``--manager``.  It binds a ``zmq.PUB`` socket on ``tcp://*:<port>`` and emits a
rolling window of ``num_frames_to_send`` frames at ``target_fps``.

This client exposes the same duck-typed surface as ``NoitomClient`` /
``XsensClient`` used by :mod:`deploy.retarget`::

    client = PicoClient(host="192.168.1.50", port=5555)
    client.start_thread()
    frame = client.get_frame_data(timeout=0.5)   # -> PicoFrame | None
    client.stop()

Unlike those two, a :class:`PicoFrame` already carries **G1 joint angles**
(``joint_pos``, 29-DoF) computed on the PICO host, so no GMR/IK is required
downstream.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from deploy.pico.protocol import (
    DEFAULT_PORT,
    DEFAULT_TOPIC,
    PicoProtocolError,
    unpack_pose_message,
)

# G1 whole-body DoF count (tracking.constants.NUM_JOINT).
NUM_JOINT = 29


@dataclass
class PicoFrame:
    """One decoded PICO frame (the newest entry of the streamed window)."""

    joint_pos: np.ndarray                      # (29,) G1 joint angles [rad]
    root_quat: np.ndarray                      # (4,) wxyz, MuJoCo convention
    smpl_pose: np.ndarray | None = None        # (21, 3) axis-angle body pose
    smpl_joints: np.ndarray | None = None      # (24, 3) local joint positions
    left_trigger: float = 0.0
    right_trigger: float = 0.0
    left_grip: float = 0.0
    right_grip: float = 0.0
    heading_increment: float = 0.0
    frame_index: int = 0
    recv_time: float = field(default_factory=time.time)
    raw: dict[str, np.ndarray] = field(default_factory=dict)


def _last(arr: np.ndarray) -> np.ndarray:
    """Take the newest frame from a ``(N, ...)`` stacked window."""
    return arr[-1] if arr.ndim > 1 else arr


def _scalar(raw: dict[str, np.ndarray], key: str, default: float = 0.0) -> float:
    val = raw.get(key)
    if val is None or val.size == 0:
        return default
    return float(np.asarray(val).reshape(-1)[-1])


class PicoClient:
    """Background-thread ZMQ SUB client holding the latest :class:`PicoFrame`."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORT,
        topic: str = DEFAULT_TOPIC,
        conflate: bool = True,
    ):
        self.host = host
        self.port = int(port)
        self.topic = topic
        self.conflate = conflate

        self._latest: PicoFrame | None = None
        self._lock = threading.Lock()
        self._new_frame = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ctx = None
        self._socket = None

        self.frames_received = 0
        self.decode_errors = 0

    # -- lifecycle ---------------------------------------------------------
    def start_thread(self) -> None:
        if self._thread is not None:
            return
        import zmq

        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.SUB)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, self.topic)
        if self.conflate:
            # Keep only the newest message: teleop wants freshness, not history.
            self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.RCVTIMEO, 200)  # ms
        self._socket.connect(f"tcp://{self.host}:{self.port}")

        self._thread = threading.Thread(
            target=self._run, name="pico-zmq-sub", daemon=True
        )
        self._thread.start()
        print(f"[PicoClient] subscribed to tcp://{self.host}:{self.port} topic={self.topic!r}")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None

    # -- consumer API ------------------------------------------------------
    def get_frame_data(self, timeout: float | bool = 0.5) -> PicoFrame | None:
        """Return the latest frame, waiting up to ``timeout`` seconds for a new one.

        ``timeout`` accepts a bool for signature-compatibility with
        ``NoitomClient.get_frame_data(timeout=True)``.
        """
        if isinstance(timeout, bool):
            timeout = 0.5 if timeout else 0.0
        if timeout > 0:
            self._new_frame.wait(timeout)
        self._new_frame.clear()
        with self._lock:
            return self._latest

    # -- worker ------------------------------------------------------------
    def _run(self) -> None:
        import zmq

        while not self._stop.is_set():
            try:
                data = self._socket.recv()
            except zmq.Again:
                continue
            except zmq.ZMQError:
                break

            try:
                raw = unpack_pose_message(data, topic=self.topic)
                frame = self._to_frame(raw)
            except (PicoProtocolError, KeyError, ValueError) as exc:
                self.decode_errors += 1
                if self.decode_errors <= 5:
                    print(f"[PicoClient] decode error: {exc}")
                continue

            with self._lock:
                self._latest = frame
            self.frames_received += 1
            self._new_frame.set()

    def _to_frame(self, raw: dict[str, np.ndarray]) -> PicoFrame:
        if "joint_pos" not in raw:
            raise KeyError(
                f"stream has no 'joint_pos' field (got {sorted(raw)}); "
                "is the publisher an older pico_manager build?"
            )

        joint_pos = np.asarray(_last(raw["joint_pos"]), dtype=np.float32).reshape(-1)
        if joint_pos.size != NUM_JOINT:
            raise ValueError(
                f"expected {NUM_JOINT} joints from PICO, got {joint_pos.size}"
            )

        if "body_quat_w" in raw:
            root_quat = np.asarray(_last(raw["body_quat_w"]), dtype=np.float32).reshape(-1)
        else:
            root_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        norm = float(np.linalg.norm(root_quat))
        root_quat = root_quat / norm if norm > 1e-6 else np.array(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float32
        )

        smpl_pose = raw.get("smpl_pose")
        smpl_joints = raw.get("smpl_joints")

        frame_index = raw.get("frame_index")
        idx = int(np.asarray(frame_index).reshape(-1)[-1]) if frame_index is not None else 0

        return PicoFrame(
            joint_pos=joint_pos,
            root_quat=root_quat.astype(np.float32),
            smpl_pose=None if smpl_pose is None else _last(smpl_pose),
            smpl_joints=None if smpl_joints is None else _last(smpl_joints),
            left_trigger=_scalar(raw, "left_trigger"),
            right_trigger=_scalar(raw, "right_trigger"),
            left_grip=_scalar(raw, "left_grip"),
            right_grip=_scalar(raw, "right_grip"),
            heading_increment=_scalar(raw, "heading_increment"),
            frame_index=idx,
            raw=raw,
        )
