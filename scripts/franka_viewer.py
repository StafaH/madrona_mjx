'''
```sh
cd mujoco_menagerie/franka_emika_panda

# Add a camera to single cube scene
cat >mjx_single_cube_camera.xml <<EOF
<mujoco model="panda scene">
  <include file="mjx_scene.xml"/>

  <worldbody>
    <camera name="front" pos="1.2 0 1" fovy="58" mode="fixed" quat="0.6532815 0.2705981 0.2705981 0.6532815 "/>
    <body name="box" pos="0.5 0 0.03">
      <freejoint/>
      <geom type="box" name="box" size="0.02 0.02 0.03" condim="3"
       friction="1 .03 .003" rgba="0 1 0 1" contype="2" conaffinity="1" solref="0.01 1"/>
    </body>
    <body mocap="true" name="mocap_target">
      <geom type="sphere" size="0.025" rgba="1 0 0 1" contype="0" conaffinity="0"/>
    </body>
  </worldbody>

  <keyframe>
    <key name="home"
      qpos="0 0.3 0 -1.57079 0 2.0 -0.7853 0.04 0.04 0.7 0 0.03 1 0 0 0"
      ctrl="0 0.3 0 -1.57079 0 2.0 -0.7853 0.04"/>
    <key name="pickup"
      qpos="0.2897 0.50732 -0.140016 -2.176 -0.0310497 2.51592 -0.49251 0.04 0.0399982 0.511684 0.0645413 0.0298665 0.665781 2.76848e-17 -2.27527e-17 -0.746147"
      ctrl="0.2897 0.423 -0.144392 -2.13105 -0.0291743 2.52586 -0.492492 0.04"/>
    <key name="pickup1"
      qpos='0.2897 0.496673 -0.142836 -2.14746 -0.0295746 2.52378 -0.492496 0.04 0.0399988 0.529553 0.0731702 0.0299388 0.94209 8.84613e-06 -4.97524e-06 -0.335361'
      ctrl="0.2897 0.458 -0.144392 -2.13105 -0.0291743 2.52586 -0.492492 0.04"/>
  </keyframe>
</mujoco>
EOF
```

Sample Command:
MADRONA_MWGPU_KERNEL_CACHE=../build/cache python franka_viewer.py --num-worlds 8 \
--window-width 2730 --window-height 1536 --batch-render-view-width 64 --batch-render-view-height 64

'''

import argparse
import functools
import os
import sys
import time
from datetime import datetime
from typing import Optional, Any, List, Sequence, Dict, Tuple, Union, Callable

# FIXME, hacky, but need to leave decent chunk of memory for Madrona /
# the batch renderer
def limit_jax_mem(limit):
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{limit:.2f}"
limit_jax_mem(0.1)

import jax
import jax.numpy as jp
import flax
import numpy as np

from brax.io import image, model, html
from brax.training.agents.ppo import train as ppo
from brax.envs import training
from brax.training.acme import running_statistics
from mujoco.mjx._src import math
from mujoco.mjx._src import io
from mujoco.mjx._src import support

from franka_env import PandaBringToTarget, RobotAutoResetWrapper
from vision_ppo import make_vision_ppo_networks, make_inference_fn

from madrona_mjx.viz import VisualizerGPUState, Visualizer

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--gpu-id', type=int, default=0)
arg_parser.add_argument('--num-worlds', type=int, required=True)
arg_parser.add_argument('--window-width', type=int, required=True)
arg_parser.add_argument('--window-height', type=int, required=True)
arg_parser.add_argument('--batch-render-view-width', type=int, required=True)
arg_parser.add_argument('--batch-render-view-height', type=int, required=True)
arg_parser.add_argument('--add-cam-debug-geo', action='store_true')
arg_parser.add_argument('--use-raytracer', action='store_true')
arg_parser.add_argument('--inference', action='store_true')
arg_parser.add_argument('--model-path', type=str, default='frankavision_model')

args = arg_parser.parse_args()

viz_gpu_state = VisualizerGPUState(args.window_width, args.window_height, args.gpu_id)

# Tell XLA to use Triton GEMM
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

if __name__ == '__main__':
  env = PandaBringToTarget(
    vision_obs=True,
    render_batch_size=args.num_worlds,
    gpu_id=args.gpu_id,
    render_width=args.batch_render_view_width,
    render_height=args.batch_render_view_height,
    add_cam_debug_geo=args.add_cam_debug_geo,
    use_rt=args.use_raytracer,
    render_viz_gpu_hdls=viz_gpu_state.get_gpu_handles(),
  )

  if args.inference:
    env = training.VmapWrapper(env)
    env = training.EpisodeWrapper(env, 500, 1)
    env = RobotAutoResetWrapper(env)
    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)
  else:
    jit_env_reset = jax.jit(jax.vmap(env.reset))
    jit_env_step = jax.jit(jax.vmap(env.step))    

  rng = jax.random.PRNGKey(seed=2)
  rng, reset_rng = jax.random.split(rng, 2)
  state = jit_env_reset(jax.random.split(reset_rng, args.num_worlds))
  
  if args.inference:
    params = model.load_params(args.model_path)
    ppo_network = make_vision_ppo_networks(
      channel_size=state.obs.shape[-1],
      action_size=env.action_size,
      preprocess_observations_fn=running_statistics.normalize)
    
    make_inference_fn = make_inference_fn(ppo_network)
    inference_fn = make_inference_fn(params)
    jit_inference_fn = jax.jit(inference_fn)

  def step_fn(carry):
    rng, state = carry

    act_rng, rng = jax.random.split(rng)
    if args.inference:
      ctrl, _ = jit_inference_fn(state.obs, act_rng)
    else:
      # ctrl = jax.random.uniform(act_rng, (args.num_worlds, env.action_size))
      ctrl = jp.zeros((args.num_worlds, env.action_size))
    state = jit_env_step(state, ctrl)

    return rng, state
    
  visualizer = Visualizer(viz_gpu_state, env.renderer.madrona)
  visualizer.loop(env.renderer.madrona, step_fn, (rng, state))
