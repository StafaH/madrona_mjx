import jax
import jax.numpy as jp
import numpy as np
import mujoco

from typing import Optional, Any, List, Sequence, Dict, Tuple, Union, Callable

from brax import base
from brax.envs.base import PipelineEnv
from brax.envs.base import State, Wrapper
from brax.io import mjcf

from etils import epath
from ml_collections import config_dict

from madrona_mjx.renderer import BatchRenderer

import os
from brax.envs import training

FRANKA_PANDA_ROOT_PATH = epath.Path('mujoco_menagerie/franka_emika_panda')

from franka_ik import compute_franka_fk, compute_franka_ik

class RobotAutoResetWrapper(Wrapper):
  """Automatically resets Brax envs that are done."""

  def reset(self, rng: jax.Array) -> State:
    state = self.env.reset(rng)
    state.info['first_pipeline_state'] = state.pipeline_state
    state.info['first_obs'] = state.obs
    state.info['first_cp'] = state.info['current_pos']
    state.info['first_reward'] = state.reward
    return state

  def step(self, state: State, action: jax.Array) -> State:
    if 'steps' in state.info:
      steps = state.info['steps']
      steps = jp.where(state.done, jp.zeros_like(steps), steps)
      state.info.update(steps=steps)
    state = state.replace(done=jp.zeros_like(state.done))
    state = self.env.step(state, action)

    def where_done(x, y):
      done = state.done
      if done.shape:
        done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
      return jp.where(done, x, y)

    pipeline_state = jax.tree.map(
        where_done, state.info['first_pipeline_state'], state.pipeline_state
    )
    obs = where_done(state.info['first_obs'], state.obs)
    state.info['current_pos'] = where_done(state.info['first_cp'], state.info['current_pos'])
    state.info['prev_reward'] = where_done(state.info['first_reward'], state.info['prev_reward'])
    return state.replace(pipeline_state=pipeline_state, obs=obs, info=state.info)


def default_config():
  """Returns reward config for the environment."""

  return config_dict.create(
      # Environment timestep. Should match the robot decision frequency.
      dt=0.02,
      # Lowers action magnitude for less-jerky motion.  Also sometimes helps
      # sample efficiency.
      action_scale=0.04,
      # The coefficients for all reward terms used for training.
      reward_scales=config_dict.create(
          # Gripper goes to the box.
          gripper_box=1.0,
          # Box goes to the target mocap.
          box_target=2.0,
          # Do not collide the gripper with the floor.
          # no_floor_collision=0.25,
          # Sparse grasp
          sparse_grasp=2.0,
      ),
  )


def _load_sys(path: epath.Path) -> base.System:
  """Load a mujoco model from a path."""
  assets = {}
  for f in path.parent.glob('*.xml'):
    assets[f.name] = f.read_bytes()
  for f in (path.parent / 'assets').glob('*'):
    assets[f.name] = f.read_bytes()
  xml = path.read_text()
  model = mujoco.MjModel.from_xml_string(xml, assets)
  return mjcf.load_model(model)


def _get_collision_info(
    contact: Any, geom1: int, geom2: int) -> Tuple[jax.Array, jax.Array]:
  if geom1 > geom2:
    geom1, geom2 = geom2, geom1
  mask = (jp.array([geom1, geom2]) == contact.geom).all(axis=1)
  idx = jp.where(mask, contact.dist, 1e4).argmin()
  dist = contact.dist[idx] * mask[idx]
  normal = (dist < 0) * contact.frame[idx, 0, :3]
  return dist, normal


def _geoms_colliding(
    state: Optional[State], geom1: int, geom2: int
) -> jax.Array:
  return _get_collision_info(state.contact, geom1, geom2)[0] < 0


class PandaBringToTarget(PipelineEnv):
  def __init__(
    self,
    vision_obs=False,
    render_batch_size=16,
    gpu_id=0,
    render_width=64,
    render_height=64,
    max_depth=3.0,
    enabled_geom_groups=np.array([0, 1, 2]),
    add_cam_debug_geo=False,
    use_rt: bool = False,
    render_viz_gpu_hdls=None,
    **kwargs):

    global root_path
    sys = _load_sys(FRANKA_PANDA_ROOT_PATH / 'mjx_single_cube_camera.xml')
    self._config = config = default_config()
    nsteps = int(np.round(config.dt / sys.opt.timestep))
    kwargs['backend'] = 'mjx'
    kwargs['n_frames'] = nsteps
    super().__init__(sys, **kwargs)

    # define constants
    self.model = sys.mj_model
    arm_joints = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5',
                  'joint6', 'joint7']
    finger_joints = ['finger_joint1', 'finger_joint2']
    all_joints = arm_joints + finger_joints
    self._robot_arm_qposadr = np.array([
        sys.mj_model.jnt_qposadr[sys.mj_model.joint(j).id] for j in arm_joints])
    self._robot_qposadr = np.array([
        sys.mj_model.jnt_qposadr[sys.mj_model.joint(j).id] for j in all_joints])
    self._gripper_site = sys.mj_model.site('gripper').id
    self._left_finger_geom = sys.mj_model.geom('left_finger_pad').id
    self._right_finger_geom = sys.mj_model.geom('right_finger_pad').id
    self._left_finger_detector_geom = sys.mj_model.geom('left_finger_pad_detector').id
    self._right_finger_detector_geom = sys.mj_model.geom('right_finger_pad_detector').id
    self._hand_geom = sys.mj_model.geom('hand_capsule').id
    self._box_body = sys.mj_model.body('box').id
    self._box_geom = sys.mj_model.geom('box').id
    self._box_qposadr = sys.mj_model.jnt_qposadr[sys.mj_model.body('box').jntadr[0]]
    # TODO(btaba): replace with mocap_pos once MJX version 3.2.3 is released.
    self._target_id = sys.mj_model.body('mocap_target').id
    self._floor_geom = sys.mj_model.geom('floor').id
    self._init_q = sys.mj_model.keyframe('home').qpos
    self._init_box_pos = jp.array(
        self._init_q[self._box_qposadr : self._box_qposadr + 3],
        dtype=jp.float32)
    self._init_ctrl = sys.mj_model.keyframe('home').ctrl
    self._lowers = sys.mj_model.actuator_ctrlrange[:, 0]
    self._uppers = sys.mj_model.actuator_ctrlrange[:, 1]
    self._max_depth = max_depth

    # Using fk ik for cartesian control
    self._start_tip_transform = compute_franka_fk(self._init_ctrl[:7])
    self._speed_multiplier = 0.005

    self._vision_obs = vision_obs
    if vision_obs:
      self.renderer = BatchRenderer(
        sys, gpu_id, render_batch_size, render_width, render_height, 
        enabled_geom_groups, add_cam_debug_geo, use_rt, render_viz_gpu_hdls)
  
  @property
  def action_size(self) -> int:
    return 3

  def _reset3d(self, rng_box: jax.Array, rng_target: jax.Array) -> jax.Array:
    # intialize box position
    box_pos = jax.random.uniform(
        rng_box, (3,),
        minval=jp.array([-0.2, -0.2, 0.0]),
        maxval=jp.array([0.2, 0.2, 0.0])) + self._init_box_pos
    target_pos = jp.array([0.5, 0.0, 0.3])
    
    return box_pos, target_pos

  def _reset2d(self, rng_box: jax.Array, rng_target: jax.Array) -> jax.Array:
    # intialize box position
    box_pos = jax.random.uniform(
        rng_box, (3,),
        minval=jp.array([0, -0.2, 0.0]),
        maxval=jp.array([0, 0.2, 0.0])) + self._init_box_pos
    box_pos = box_pos.at[0].set(self._start_tip_transform[0, 3])
    target_pos = jp.array([0.5, 0.0, 0.3])
    
    return box_pos, target_pos

  def reset(self, rng: jax.Array) -> State:
    """Resets the environment to an initial state."""
    rng, rng_box, rng_target = jax.random.split(rng, 3)

    box_pos, target_pos = self._reset2d(rng_box, rng_target)
    
    # initialize pipeline state
    init_q = jp.array(self._init_q).at[
        self._box_qposadr : self._box_qposadr + 3].set(box_pos)
    pipeline_state = self.pipeline_init(
        init_q, jp.zeros(self.sys.nv)
    )
    pipeline_state = pipeline_state.replace(ctrl=self._init_ctrl)
    # set target mocap position
    # TODO(btaba): replace with mocap_pos once MJX version 3.2.3 is released.
    # pipeline_state = pipeline_state.replace(
    #     xpos=pipeline_state.xpos.at[self._target_id, :].set(target_pos))

    # initialize env state and info
    metrics = {
        'out_of_bounds': jp.array(0.0),
        **{k: 0.0 for k in self._config.reward_scales.keys()},
    }
    info = {'rng': rng,
            'target_pos': target_pos,
            'reached_box': 0.0,
            'prev_reward': 0.0,
            'current_step': 0,
            'current_pos': self._start_tip_transform[:3, 3]}
    reward, done = jp.zeros(2)

    if self._vision_obs:
      render_token, rgb, depth = self.renderer.init(pipeline_state)
      obs = jp.asarray(rgb[0], dtype=jp.float32) / 255.0
      norm_depth = jp.clip(depth[0], 0, self._max_depth)
      norm_depth = norm_depth / self._max_depth
      obs = obs.at[:, :, 3].set(norm_depth[..., 0])
      info.update({
        'render_token': render_token,
        'rgb': rgb[0],
        'depth': depth[0],
      })
    else:
      obs = self._get_obs(pipeline_state, info)

    return State(pipeline_state, obs, reward, done, metrics, info)

  def _step3d(self, state: State, action: jax.Array) -> jax.Array:
    return action

  def _step2d(self, state: State, action: jax.Array) -> jax.Array:
    mod_action = jp.zeros(4)
    mod_action = mod_action.at[1:4].set(action)
    return mod_action

  def step(self, state: State, action: jax.Array) -> State:
    """Runs one timestep of the environment's dynamics."""
    action = self._step2d(state, action)
    ctrl, new_tip_position = self._move_tip(
      state.info['current_pos'],
      self._start_tip_transform[:3, :3],
      state.pipeline_state.qpos,
      action,
      self._speed_multiplier)
    ctrl = jp.clip(ctrl, self._lowers, self._uppers)

    # step the physics
    data = self.pipeline_step(state.pipeline_state, ctrl)

    state.info.update({'current_pos': new_tip_position})

    # compute reward terms
    target_pos = state.info['target_pos']
    box_pos = data.xpos[self._box_body]
    gripper_pos = data.site_xpos[self._gripper_site]
    box_target = self._exp_distance(box_pos, target_pos, 0.02, 100.0, 5.0)
    gripper_box = self._exp_distance(gripper_pos, box_pos, 0.02, 100.0, 5.0)

    finger_box_collision = [
        _geoms_colliding(state.pipeline_state, self._box_geom, g)
        for g in [self._left_finger_detector_geom, self._right_finger_detector_geom]
    ]

    box_grasped = sum(finger_box_collision) > 1

    rewards = {
        'box_target': box_target * box_grasped,
        'gripper_box': gripper_box,
        'sparse_grasp': box_grasped,
    }
    rewards = {k: v * self._config.reward_scales[k] for k, v in rewards.items()}
    total_reward = jp.clip(sum(rewards.values()), -1e4, 1e4)
    total_reward = jp.where(jp.isnan(total_reward), 0.0, total_reward)

    reward = total_reward - state.info['prev_reward']
    state.info['prev_reward'] = total_reward

    out_of_bounds = jp.any(jp.abs(box_pos) > 1.0)
    out_of_bounds |= box_pos[2] < 0.0

    done = out_of_bounds | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    done = done.astype(float)
    
    if self._vision_obs:
      _, rgb, depth = self.renderer.render(state.info['render_token'], data)
      state.info.update({'rgb': rgb[0], 'depth': depth[0]})
      obs = jp.asarray(rgb[0], dtype=jp.float32) / 255.0
      norm_depth = jp.clip(depth[0], 0, self._max_depth)
      norm_depth = norm_depth / self._max_depth
      obs = obs.at[:, :, 3].set(norm_depth[..., 0])
    else:
      obs = self._get_obs(data, state.info)

    state.metrics.update(
        out_of_bounds=out_of_bounds.astype(float),
        **rewards)

    return state.replace(
      pipeline_state=data, obs=obs, reward=reward, done=done, info=state.info)
  
  def _get_obs(self, data: State, info: dict[str, Any]) -> jax.Array:
    gripper_pos = data.site_xpos[self._gripper_site]
    gripper_mat = data.site_xmat[self._gripper_site].ravel()
    obs = jp.concatenate([
        data.qpos,
        data.qvel,
        gripper_pos,
        gripper_mat[3:],
        data.xmat[self._box_body].ravel()[3:],
        data.xpos[self._box_body] - data.site_xpos[self._gripper_site],
        info['target_pos'] - data.xpos[self._box_body],
        data.ctrl - data.qpos[self._robot_qposadr[:-1]],
    ])
    return obs

  def _move_tip(
    self,
    current_tip_pos: jax.Array,
    current_tip_rot: jax.Array,
    current_qpos: jax.Array,
    action: jax.Array,
    speed_multiplier: float) -> jax.Array:
    """Calculate new tip position from cartesian increment."""
    scaled_pos = action[:3] * speed_multiplier
    new_tip_pos = current_tip_pos.at[:3].add(scaled_pos)

    new_jp = current_qpos[:8]
  
    new_tip_pos = new_tip_pos.at[0].set(jp.clip(new_tip_pos[0], 0.25, 0.77))
    new_tip_pos = new_tip_pos.at[1].set(jp.clip(new_tip_pos[1], -0.32, 0.32))
    new_tip_pos = new_tip_pos.at[2].set(jp.clip(new_tip_pos[2], 0.02, 0.5))

    new_tip_mat = jp.identity(4)
    new_tip_mat = new_tip_mat.at[:3, :3].set(current_tip_rot)
    new_tip_mat = new_tip_mat.at[:3, 3].set(new_tip_pos)

    out_jp = compute_franka_ik(new_tip_mat, current_qpos[6], current_qpos[:7])
    out_jp = jp.where(jp.any(jp.isnan(out_jp)), current_qpos[:7], out_jp)
    new_tip_pos = jp.where(jp.any(jp.isnan(new_jp)), current_tip_pos, new_tip_pos)
    
    new_jp = new_jp.at[:7].set(out_jp)
    new_jp = new_jp.at[8].set(jp.clip(action[3], 0.0, 0.04))

    return new_jp, new_tip_pos
  
  def _exp_distance(
    self,
    cur_pos: jax.Array,
    tar_pos: jax.Array,
    tol: float,
    max_dist: float,
    exp_multiplier: float) -> jax.Array:
    """Exponential shaped distance reward function."""
    d = jp.linalg.norm(jp.asarray(cur_pos - tar_pos), ord=2)
    d -= tol
    d = jp.clip(d, min=0, max=max_dist)
    return jp.exp(-exp_multiplier * d)

if __name__ == '__main__':

  num_worlds = 8
  def limit_jax_mem(limit):
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{limit:.2f}"
  
  limit_jax_mem(0.2)
  env = PandaBringToTarget(
    vision_obs=False,
    render_batch_size=num_worlds,
    gpu_id=0,
    render_width=64,
    render_height=64,
    use_rt=False)

  env = training.VmapWrapper(env)
  env = training.EpisodeWrapper(env, 20, 1)
  env = RobotAutoResetWrapper(env)

  reset_fn = jax.jit(env.reset)
  step_fn = jax.jit(env.step)

  rng = jax.random.PRNGKey(0)
  rng, reset_rng = jax.random.split(rng, 2)
  mjx_state = reset_fn(jax.random.split(reset_rng, num_worlds))

  for i in range(5):
    act = jp.ones((num_worlds, env.action_size))
    mjx_state = step_fn(mjx_state, act * 0.005)
  
  print("Success!")