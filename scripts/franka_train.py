'''
Instructions:

1. Build madrona from new_renderer pip install
2. pip install brax from source (required for wrap_env, which is unreleased)
3. Add mjx_single_cube_camera.xml to mujoco_menagerie
4. Run this script from this directory (due to relative paths for mujoco menagerie and build)

```sh
cd mujoco_menagerie/franka_emika_panda

# Add a camera to single cube scene
cat >mjx_single_cube_camera.xml <<EOF
<mujoco model="panda scene">
  <include file="mjx_scene.xml"/>

  <worldbody>
    <camera name="front" pos="1.3 0 0.6" fovy="58" mode="fixed" euler="0 1.2 1.5708"/>
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
      qpos="0 0.3 0 -1.57079 0 1.94 0.785 0.04 0.04 0.7 0 0.03 1 0 0 0"
      ctrl="0 0.3 0 -1.57079 0 1.94 0.785 0.04"/>
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
MADRONA_MWGPU_KERNEL_CACHE=../build/cache python franka_train.py \
  --num-worlds 1024 --batch-render-view-width 64 --batch-render-view-height 64 \
  --num-steps 20000000 --save-model --save-path frankavision_model
'''

import argparse
import os
import time
import functools
from matplotlib import pyplot as plt

import jax
from brax.envs import training
from brax.io import model
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks

from franka_env import PandaBringToTarget, RobotAutoResetWrapper
from vision_ppo import make_vision_ppo_networks

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--vision', action='store_true')
arg_parser.add_argument('--num-worlds', type=int, required=True)
arg_parser.add_argument('--num-steps', type=int, default=20_000_000)
arg_parser.add_argument('--save-model', action='store_true')
arg_parser.add_argument('--save-path', type=str, default='franka_model')

# Vision Args
arg_parser.add_argument('--gpu-id', type=int, default=0)
arg_parser.add_argument('--batch-render-view-width', type=int, default=64)
arg_parser.add_argument('--batch-render-view-height', type=int, default=64)
arg_parser.add_argument('--use-raytracer', action='store_true')

args = arg_parser.parse_args()

def limit_jax_mem(limit):
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{limit:.2f}"
limit_jax_mem(0.65)


# Tell XLA to use Triton GEMM
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags


if __name__ == '__main__':
  print("Initializing...")
  env = PandaBringToTarget(
    vision_obs=args.vision,
    render_batch_size=args.num_worlds,
    gpu_id=args.gpu_id,
    render_width=args.batch_render_view_width,
    render_height=args.batch_render_view_height,
    use_rt=args.use_raytracer,
    max_depth=2)
  
  episode_length = 500
  action_repeat = 2

  if args.vision:
    network_factory = functools.partial(
      make_vision_ppo_networks,
      policy_hidden_layer_sizes=[128, 128, 128],
      value_hidden_layer_sizes=[128, 128, 128],
      image_dim=(args.batch_render_view_width, args.batch_render_view_height))
    num_eval_envs = args.num_worlds
    batch_size = 256
  else:
    network_factory = functools.partial(
      ppo_networks.make_ppo_networks,
          policy_hidden_layer_sizes=(32, 32, 32, 32))
    num_eval_envs = 128
    batch_size = 1024
    num_minibatches = 32
    num_updates_per_batch = 8
    discounting=0.97
    learning_rate=1e-3
    entropy_cost=2e-2
    discounting=0.97

  env = training.VmapWrapper(env)
  env = training.EpisodeWrapper(env, episode_length=episode_length, action_repeat=action_repeat)
  env = RobotAutoResetWrapper(env)

  train_fn = functools.partial(
    ppo.train, num_timesteps=args.num_steps, num_evals=5, reward_scaling=1.0,
    episode_length=episode_length, normalize_observations=True, action_repeat=action_repeat,
    unroll_length=10, num_minibatches=num_minibatches, num_updates_per_batch=num_updates_per_batch,
    discounting=discounting, learning_rate=learning_rate, entropy_cost=entropy_cost, 
    num_envs=args.num_worlds, num_eval_envs=num_eval_envs, num_resets_per_eval=1,
    batch_size=batch_size, seed=0, network_factory=network_factory, wrap_env=False)


  def progress(num_steps, metrics):
    print(f'step: {num_steps}, reward: {metrics["eval/episode_reward"]}')

  print("Starting training...")
  start = time.time()
  make_inference_fn, params, metrics = train_fn(
    environment=env, progress_fn=progress)
  end = time.time()
  train_time = end - start

  print(
    f"""
    Summary for gpu training
    Total simulation time: {train_time:.2f} s
    Total training wall time: {metrics['training/walltime']} s
    Total eval wall time: {metrics['eval/walltime']} s
    Total time per step: { 1e6 * train_time / args.num_steps:.2f} µs""")

  if args.save_model:
    model.save_params(args.save_path, params)