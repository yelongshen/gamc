import numpy as np
try:
    from smplx import create
    model = create('/home/grease/miniforge3/envs/h-gpt/lib/python3.12/site-packages/smplx/body_models.py', model_type='smpl', ext='npz')
    print("smplx successfully loaded model.")
except Exception as e:
    print(f"Error: {e}")
