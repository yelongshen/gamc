"""Mock PICO publisher — lets you debug the sim deploy path with no headset.

Emits the same ZMQ PUB wire format as
``gear_sonic/scripts/pico_manager_thread_server.py --manager``, filled with a
synthetic waving/squatting motion so the whole chain
(``PicoClient`` -> ``deploy.retarget`` -> ``play_track`` mode 1) can be
exercised offline.

Usage::

    # terminal 1
    python -m deploy.pico.mock_server --port 5555 --fps 50

    # terminal 2
    python -m deploy.play_track --mocap-type pico --pico-host 127.0.0.1
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import tyro

from deploy.pico.protocol import DEFAULT_PORT, DEFAULT_TOPIC, pack_pose_message

NUM_JOINT = 29

# Indices into the G1 29-DoF vector that we animate (shoulder pitch / elbow).
_L_SHOULDER_PITCH = 15
_L_ELBOW = 18
_R_SHOULDER_PITCH = 22
_R_ELBOW = 25


@dataclass
class MockArgs:
    """Publish a synthetic PICO SMPL/G1 pose stream for offline debugging."""

    port: int = DEFAULT_PORT
    topic: str = DEFAULT_TOPIC
    fps: int = 50
    """Publish rate (matches pico_manager --target_fps)."""
    num_frames_to_send: int = 5
    """Frames per message (matches pico_manager --num_frames_to_send)."""
    motion: str = "wave"
    """`wave` (arms), `squat` (knees), or `still`."""
    amplitude: float = 0.6
    """Peak joint excursion in radians."""


def _joint_pos(t: float, motion: str, amp: float) -> np.ndarray:
    q = np.zeros(NUM_JOINT, dtype=np.float32)
    s = np.sin(2.0 * np.pi * 0.5 * t)

    if motion == "wave":
        q[_L_SHOULDER_PITCH] = -amp * (0.5 + 0.5 * s)
        q[_R_SHOULDER_PITCH] = -amp * (0.5 - 0.5 * s)
        q[_L_ELBOW] = 0.5 * amp * (1.0 + s)
        q[_R_ELBOW] = 0.5 * amp * (1.0 - s)
    elif motion == "squat":
        # hip pitch / knee / ankle pitch for both legs
        for hip, knee, ankle in ((0, 3, 4), (6, 9, 10)):
            q[hip] = -0.5 * amp * (1.0 + s)
            q[knee] = amp * (1.0 + s)
            q[ankle] = -0.5 * amp * (1.0 + s)
    return q


def main(args: MockArgs) -> None:
    import zmq

    ctx = zmq.Context()
    socket = ctx.socket(zmq.PUB)
    socket.bind(f"tcp://*:{args.port}")
    print(f"[MockPico] PUB on tcp://*:{args.port} topic={args.topic!r} "
          f"motion={args.motion} fps={args.fps}")
    print("[MockPico] waiting 0.5 s for subscribers to attach...")
    time.sleep(0.5)

    dt = 1.0 / max(1, args.fps)
    N = args.num_frames_to_send
    step = 0
    t0 = time.time()

    try:
        while True:
            t = time.time() - t0

            # Rolling window of N frames, oldest first.
            joint_pos = np.stack(
                [_joint_pos(t - (N - 1 - i) * dt, args.motion, args.amplitude)
                 for i in range(N)],
                axis=0,
            )
            body_quat = np.tile(
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (N, 1)
            )

            payload = {
                "smpl_pose": np.zeros((N, 21, 3), dtype=np.float32),
                "smpl_joints": np.zeros((N, 24, 3), dtype=np.float32),
                "body_quat_w": body_quat,
                "joint_pos": joint_pos.astype(np.float32),
                "joint_vel": np.zeros((N, NUM_JOINT), dtype=np.float32),
                "frame_index": np.arange(step, step + N, dtype=np.int64),
                "left_trigger": np.array([0.0], dtype=np.float32),
                "right_trigger": np.array([0.0], dtype=np.float32),
                "left_grip": np.array([0.0], dtype=np.float32),
                "right_grip": np.array([0.0], dtype=np.float32),
                "heading_increment": np.array([0.0], dtype=np.float32),
                "pico_fps": np.array([float(args.fps)], dtype=np.float32),
                "timestamp_realtime": np.array([time.time()], dtype=np.float64),
            }

            socket.send(pack_pose_message(payload, topic=args.topic))
            step += 1

            if step % (args.fps * 5) == 0:
                print(f"[MockPico] sent {step} messages ({t:.0f}s)")

            time.sleep(dt)
    except KeyboardInterrupt:
        print("\n[MockPico] stopped")
    finally:
        socket.close(linger=0)
        ctx.term()


if __name__ == "__main__":
    main(tyro.cli(MockArgs))
