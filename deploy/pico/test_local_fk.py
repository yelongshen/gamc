import torch
import numpy as np

def demo_direct_fk():
    import sys
    sys.path.append("/home/grease/gam/gear_sonic")
    from trl.utils.torch_transform import compute_human_joints, angle_axis_to_quaternion, quaternion_to_rotation_matrix, quat_apply
    print("Imports worked!")

demo_direct_fk()
