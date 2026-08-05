"""Minimal tracking policy deployment script for simulation via DDS.

This reads PICO mocap frames directly via ZMQ, runs the ONNX tracking policy,
and communicates with the `run_sim_loop` emulator over DDS loopback (`lo`).
Bypasses the `play_track` keyboard and walk modes completely.
"""

from __future__ import annotations

import signal
import sys
import time
from dataclasses import dataclass

import mujoco
import numpy as np
import tyro
import pickle
from scipy.spatial.transform import Rotation as R

from deploy.pico.client import PicoClient
from deploy.real_robot import LowLevelControlG1, KeyMap
from tracking import constants as consts
from tracking.constants import DEFAULT_QPOS_JOINT
from tracking.infer_utils import G1TrackInferFn, g1_infer_env_config
from tracking.policy import Args as PolicyArgs
from tracking.policy import get_policy_onnx
from utils.ref_ghost import LiveRefConverter

try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.utils.thread import RecurrentThread
except ImportError:
    print("FATAL: unitree_sdk2py not found. See deploy/DEPLOY.md.", file=sys.stderr)
    sys.exit(1)


@dataclass
class PicoTrackArgs:
    """Minimal standalone tracking policy wrapper."""
    
    # Model
    onnx_track: str = "storage/ckpts/pns_wo_priv216.onnx"
    convert_xml_path: str = str(consts.TRACK_XML)
    
    # DDS
    net: str = "lo"            # Default to loopback for sim deployment
    freq: int = 50
    debug: bool = False        # Set true to inhibit publishing
    
    # PICO
    pico_host: str = "127.0.0.1"
    pico_port: int = 5555
    pico_topic: str = "pose"
    pico_root_height: float = 0.793
    actual_human_height: float = 1.75
    
    # Engine
    use_trt: bool = False      # Default false in sim


SMPL_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2", "L_Ankle", "R_Ankle", 
    "Spine3", "L_Foot", "R_Foot", "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", 
    "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand"
]
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

class SMPLForwardKinematics:
    def __init__(self, pkl_path="/home/grease/gam/gear_sonic/data/human/human_joints_info.pkl"):
        sys.path.append("/home/grease/gam/gear_sonic")
        with open(pkl_path, "rb") as f:
            info = pickle.load(f)
        
        self.J = info['J'][:24].numpy()
        self.parents = SMPL_PARENTS
        self.local_vec = np.zeros((24, 3))
        for i in range(1, 24):
            self.local_vec[i] = self.J[i] - self.J[self.parents[i]]
            
    def get_global_poses(self, root_quat, smpl_pose, root_translation):
        """
        root_quat: [w, x, y, z]
        smpl_pose: [21, 3] local axis-angles
        root_translation: [3,]
        """
        G_pos = np.zeros((24, 3))
        
        # Scipy uses xyzw
        r_rot = R.from_quat([root_quat[1], root_quat[2], root_quat[3], root_quat[0]])
        G_pos[0] = root_translation
        scipy_rots = [r_rot]
        
        for i in range(1, 24):
            p = self.parents[i]
            if i - 1 < len(smpl_pose):
                aa = smpl_pose[i - 1]
            else:
                aa = np.zeros(3)
            
            mag = np.linalg.norm(aa)
            if mag > 1e-6:
                local_rot = R.from_rotvec(aa)
            else:
                local_rot = R.from_quat([0, 0, 0, 1])
                
            global_rot = scipy_rots[p] * local_rot
            scipy_rots.append(global_rot)
            G_pos[i] = G_pos[p] + scipy_rots[p].apply(self.local_vec[i])
        
        out = {}
        for i in range(22):
            qx, qy, qz, qw = scipy_rots[i].as_quat()
            out[SMPL_NAMES[i]] = [G_pos[i], np.array([qw, qx, qy, qz])]
        return out


def main(args: PicoTrackArgs):
    ctrl_dt = 1.0 / args.freq
    env_cfg = g1_infer_env_config(ctrl_dt=ctrl_dt)

    # 1. Initialize SMPL FK and GMR Retargeting
    print("Initializing SMPL FK & GMR...")
    sys.path.append("/home/grease/gam/gear_sonic")
    sys.path.append("/home/grease/gamc")
    
    from gmr.params import IK_CONFIG_DICT, IK_CONFIG_ROOT
    IK_CONFIG_DICT["smpl"] = {"unitree_g1": IK_CONFIG_ROOT / "smpl_to_g1.json"}
    from gmr.motion_retarget import GeneralMotionRetargeting
    
    fk = SMPLForwardKinematics()
    retargeter = GeneralMotionRetargeting(
        src_human="smpl",
        tgt_robot="unitree_g1",
        actual_human_height=args.actual_human_height
    )

    # 2. Load Policy
    print(f"Loading policy from {args.onnx_track} (TRT={args.use_trt})")
    policy_args = PolicyArgs(onnx_track=args.onnx_track)
    track_policy = get_policy_onnx(policy_args, use_trt=args.use_trt, strict_trt=args.use_trt)

    # 2. Setup MuJoCo (FK only)
    xml_path = args.convert_xml_path
    phantom_model = mujoco.MjModel.from_xml_path(xml_path)
    phantom_model.opt.timestep = 0.001
    infer_fn = G1TrackInferFn(env_cfg, phantom_model, track_policy, privileged=False)
    live_converter = LiveRefConverter(phantom_model, ctrl_dt)

    # 3. Connect PICO Client
    print(f"Connecting to PICO stream on {args.pico_host}:{args.pico_port}")
    mocap_client = PicoClient(host=args.pico_host, port=args.pico_port, topic=args.pico_topic)
    mocap_client.start_thread()

    # 4. Init DDS
    print(f"Initializing DDS on interface `{args.net}`...")
    ChannelFactoryInitialize(0, args.net)
    low_ctrl = LowLevelControlG1(ctrl_dt=ctrl_dt, debug=args.debug)

    print("Network initialized. Validating PICO stream...")
    # Wait for the first PICO frame so we don't start the robot blind
    for _ in range(30):
        if mocap_client.get_frame_data(timeout=0.2) is not None:
            break
    else:
        print("WARNING: No PICO data received. Tracking will remain stationary until data arrives.")

    # 5. Startup sequence (matches real robot)
    print("\n<Mode: Damping> Waiting for <start> on remote/emulator...")
    while low_ctrl.remote.button[KeyMap.start] != 1:
        low_ctrl.set_motor_damping()
        time.sleep(ctrl_dt)

    low_ctrl.move_to_default_pos(duration=2.0)
    print("\n<Mode: Default> Waiting for <A> on remote/emulator...")
    while low_ctrl.remote.button[KeyMap.A] != 1:
        low_ctrl.step(DEFAULT_QPOS_JOINT, consts.KPs, consts.KDs)
        time.sleep(ctrl_dt)

    # 6. Control Loop
    prev_online_ref = None

    def locomotion_step():
        nonlocal prev_online_ref
        
        # Read robot state
        root_quat, root_gyro, jnt_qpos, jnt_qvel = low_ctrl.get_sensor_state()
        
        # Read Mocap
        frame = mocap_client.get_frame_data(timeout=0.0)
        
        if frame is not None and frame.smpl_pose is not None:
            # PICO stream FK -> GMR Dict
            root_trans = np.array([0, 0, args.pico_root_height])
            human_dict = fk.get_global_poses(frame.root_quat, frame.smpl_pose[:21], root_trans)
            
            # Perform IK with GMR to solve for 29 joints + body orientation
            try:
                qpos_full = retargeter.retarget(human_dict)
            except Exception as e:
                print(f"[IK Error]: {e}")
                qpos_full = None

            if qpos_full is not None:
                ref_new = live_converter.convert(qpos_full)
                if prev_online_ref is None:
                    # Initialize state
                    infer_fn.info["last_action"][:] = 0
                    live_converter.set_robot_initial_pose(root_quat, np.array([0.0, 0.0], dtype=float))
                    ref_curr = ref_new
                else:
                    ref_curr = prev_online_ref
            
                ref_next = ref_new
                prev_online_ref = ref_new
                
                # Run inference
                motor_targets = infer_fn.infer_onnx_real(
                    root_quat, root_gyro, jnt_qpos, jnt_qvel,
                    {"ref_curr": ref_curr, "ref_next": ref_next},
                )
                low_ctrl.step(np.asarray(motor_targets).flatten(), consts.KPs, consts.KDs)
        else:
            # Dropouts handled loosely in a minimal fashion: apply damping
            # or keep last target. Here, holding last PD is safer for stationary.
            # Real hardware paths might use advanced fallbacks.
            pass

    print("\n<Mode: Tracking> Starting inference loop...")
    loco_thread = RecurrentThread(interval=ctrl_dt, target=locomotion_step, name="loco")
    loco_thread.Start()

    # Wait for Exit
    running = True

    def _sigint(*_):
        nonlocal running
        print("Shutting down...")
        running = False

    signal.signal(signal.SIGINT, _sigint)
    
    try:
        while running:
            time.sleep(1.0)
    finally:
        loco_thread.Stop()
        mocap_client.stop()
        sys.exit(0)


if __name__ == "__main__":
    main(tyro.cli(PicoTrackArgs))
