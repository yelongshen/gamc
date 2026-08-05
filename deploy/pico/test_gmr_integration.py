import sys
import numpy as np

# 1. Modify sys path to find both repo requirements
sys.path.append("/home/grease/gam/gear_sonic")
sys.path.append("/home/grease/gamc")

# Test to make sure the pipeline builds successfully
from gmr.motion_retarget import GeneralMotionRetargeting

class MockSMPLRetarget:
    def __init__(self):
        # 1. Initialize GMR IK solver against the our new smpl mapping
        from gmr.params import IK_CONFIG_DICT, IK_CONFIG_ROOT
        IK_CONFIG_DICT["smpl"] = {"unitree_g1": IK_CONFIG_ROOT / "smpl_to_g1.json"}
        
        self.retarget = GeneralMotionRetargeting(
            src_human="smpl",
            tgt_robot="unitree_g1",
            actual_human_height=1.75
        )
        print("Success! GMR loaded our custom smpl_to_g1 profile!")

run = MockSMPLRetarget()
