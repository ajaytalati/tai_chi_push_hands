"""

See - https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/mjx/tutorial.ipynb

Use conda env mujoco-mjx

"""

import os

# Ensure GLFW is used instead of OSMesa if possible - must be set before importing dm_control
os.environ["MUJOCO_GL"] = "glfw"

import jax
print(jax.devices())

# if the print out is not [CudaDevice(id=0)] need to restart machine !!!
# %%

#@title Import packages for plotting and creating graphics
import time
import itertools
import numpy as np
from typing import Callable, NamedTuple, Optional, Union, List

# Graphics and plotting.
#print('Installing mediapy:')
#command -v ffmpeg >/dev/null || (apt update && apt install -y ffmpeg)
#pip install mediapy
import mediapy as media
import matplotlib.pyplot as plt

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)

#@title Import MuJoCo, MJX, and Brax
from datetime import datetime
from etils import epath
import functools
from IPython.display import HTML
from typing import Any, Dict, Sequence, Tuple, Union
import os
from ml_collections import config_dict

import jax
from jax import numpy as jp
import numpy as np
from flax.training import orbax_utils
from flax import struct
from matplotlib import pyplot as plt
import mediapy as media
from orbax import checkpoint as ocp

import mujoco
from mujoco import mjx

from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model

# %% Define Custom State Class - since MJX/JAX is stateless, you must explicitly manage historical state through a custom state class

"""
from flax import struct

@struct.dataclass
class CustomEnvState:
    pipeline_state: mjx.Data
    obs: jp.ndarray
    reward: jp.ndarray
    done: jp.ndarray
    metrics: dict
    pendulum_pos_history: jp.ndarray  # shape (window_size, 3)
    info: Dict[str, Any]  # required for compatibility with Brax wrappers
"""

import flax.struct
from brax.envs import State as BraxState
import jax.numpy as jp

@flax.struct.dataclass
class CustomEnvState(BraxState):
    pendulum_pos_history: jp.ndarray = flax.struct.field(
        default_factory=lambda: jp.zeros((20, 3))
    )

# %% Define Humanoid Env
# Define Humanoid Environment Root Path
HUMANOID_ROOT_PATH = epath.Path("/home/ajay/Python_Projects/mujoco-mjx/mjx/mujoco/mjx/test_data/humanoid")

class Humanoid(PipelineEnv):
    def __init__(
        self,
        ctrl_cost_weight=0.1,
        healthy_reward=5.0,
        contact_weight=0.2,
        terminate_when_unhealthy=True,
        healthy_z_range=(1.0, 2.0),
        reset_noise_scale=1e-2,
        exclude_current_positions_from_observation=False,
        **kwargs,
    ):
        # Load MJX model
        mj_model = mujoco.MjModel.from_xml_path(
            (HUMANOID_ROOT_PATH / "humanoid_with_punchbag_pendulum.xml").as_posix()
        )
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6

        sys = mjcf.load_model(mj_model)

        # Define physics steps per control step
        physics_steps_per_control_step = 5
        kwargs["n_frames"] = kwargs.get("n_frames", physics_steps_per_control_step)
        kwargs["backend"] = "mjx"
        
        super().__init__(sys, **kwargs)
        
        # Store geometry IDs during initialization
        self.punchbag_geom_id = mj_model.geom("punchbag").id
        self.punchbag_body_id = mj_model.body("punchbag_body").id
        self.hand_right_geom_id = mj_model.geom("hand_right").id
        self.hand_left_geom_id = mj_model.geom("hand_left").id
        self.forearm_right_geom_id = mj_model.geom("lower_arm_right").id
        self.forearm_left_geom_id = mj_model.geom("lower_arm_left").id

        # Environment parameters
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_weight = contact_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation

    def get_arm_touch_obs(self, data: mjx.Data) -> jp.ndarray:
        """
        Returns binary indicators (0 or 1) if left/right hands or forearms are touching the punchbag.
        """
    
        contacts = data.contact
        geom1_ids = contacts.geom1
        geom2_ids = contacts.geom2
    
        punchbag_geom_id = self.punchbag_geom_id
    
        # Define all geoms for each arm side
        right_arm_geoms = jp.array([self.hand_right_geom_id, self.forearm_right_geom_id])
        left_arm_geoms = jp.array([self.hand_left_geom_id, self.forearm_left_geom_id])
    
        # Check right arm contacts
        right_arm_contact = jp.any(
            jp.logical_or(
                jp.logical_and(geom1_ids == punchbag_geom_id, jp.any(geom2_ids[:, None] == right_arm_geoms, axis=1)),
                jp.logical_and(geom2_ids == punchbag_geom_id, jp.any(geom1_ids[:, None] == right_arm_geoms, axis=1))
            )
        )
    
        # Check left arm contacts
        left_arm_contact = jp.any(
            jp.logical_or(
                jp.logical_and(geom1_ids == punchbag_geom_id, jp.any(geom2_ids[:, None] == left_arm_geoms, axis=1)),
                jp.logical_and(geom2_ids == punchbag_geom_id, jp.any(geom1_ids[:, None] == left_arm_geoms, axis=1))
            )
        )
        
        # Return binary indicators as floats (0.0 or 1.0)
        return jp.array([right_arm_contact, left_arm_contact], dtype=jp.float32)


    def reward_arm_contact(self, state):
        """
        Compute reward for maintaining contact between the humanoid's hands or forearms and the ball.
        JAX-compatible implementation.
        """
        contacts = state.pipeline_state.contact
    
        # Extract geom IDs from contacts
        geom1_ids = contacts.geom1
        geom2_ids = contacts.geom2
    
        # punchbag geometry
        punchbag_geom_id = self.punchbag_geom_id
    
        # Arm geometry IDs (hands and forearms)
        arm_geom_ids = jp.array([
            self.hand_right_geom_id, 
            self.hand_left_geom_id,
            self.forearm_right_geom_id,
            self.forearm_left_geom_id
        ])
    
        # Check if any arm geom is in contact with ball geom
        is_punchbag_arm_contact = jp.logical_or(
            jp.logical_and(
                geom1_ids == punchbag_geom_id,
                jp.any(geom2_ids[:, None] == arm_geom_ids, axis=1)
            ),
            jp.logical_and(
                geom2_ids == punchbag_geom_id,
                jp.any(geom1_ids[:, None] == arm_geom_ids, axis=1)
            )
        )
    
        # Reward: +1 if any arm geom contacts the ball, else 0
        reward = jp.where(jp.any(is_punchbag_arm_contact), 1.0, 0.0)
    
        return reward

    def compute_plane_deviation_reward(self, pendulum_pos_history: jp.ndarray) -> jp.ndarray:
        # Remove the mean position
        pos_mean = jp.mean(pendulum_pos_history, axis=0)
        centered_positions = pendulum_pos_history - pos_mean
    
        # Perform SVD to find the plane of best fit
        u, s, vh = jp.linalg.svd(centered_positions, full_matrices=False)
        normal_vector = vh[-1, :]  # Normal vector to the best-fit plane is last singular vector
    
        # Measure deviation as recent point distance from the plane (signed distance)
        recent_pos = pendulum_pos_history[-1] - pos_mean
        deviation = jp.abs(jp.dot(recent_pos, normal_vector))
    
        # Reward proportional to deviation (scaled suitably)
        reward = deviation * 10.0  # adjust scale as desired
    
        return reward

    def reset(self, rng: jp.ndarray) -> CustomEnvState:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)
        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        
        qpos = self.sys.qpos0 + jax.random.uniform(rng1, (self.sys.nq,), minval=low, maxval=hi)
        qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=low, maxval=hi)
        
        data = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(data, jp.zeros(self.sys.nu))
        
        reward, done, zero = jp.zeros(3)
        metrics = {'reward': zero}
        
        pendulum_pos_history = jp.zeros((20, 3))  # initialize history buffer
        
        return CustomEnvState(
            pipeline_state=data,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            pendulum_pos_history=pendulum_pos_history
        )

    def step(self, state: State, action: jp.ndarray) -> CustomEnvState:
        """Runs one timestep of the environment's dynamics."""
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)

        #------
        # Obtain the current position ofpunchbag or pendulum ball
        pendulum_pos = data.xpos[self.punchbag_body_id]  # assuming you saved this ID earlier
        
        # Shift the history window and append the current position
        pendulum_pos_history = jp.roll(state.pendulum_pos_history, shift=-1, axis=0)
        pendulum_pos_history = pendulum_pos_history.at[-1].set(pendulum_pos)
        
        # The reward for moving opponents CoM
        plane_reward = self.compute_plane_deviation_reward(pendulum_pos_history)
        #------
        
        min_z, max_z = self._healthy_z_range
        is_healthy = jp.where(data.q[2] < min_z, 0.0, 1.0)
        is_healthy = jp.where(data.q[2] > max_z, 0.0, is_healthy)

        healthy_reward = self._healthy_reward if self._terminate_when_unhealthy else self._healthy_reward * is_healthy
        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

        obs = self._get_obs(data, action)  
        
        reward = healthy_reward - ctrl_cost + self._contact_weight * self.reward_arm_contact(state) + 5*plane_reward
        
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
        
        metrics = {'reward': reward}
            
        return state.replace(
            pipeline_state=data, obs=obs, reward=reward, done=done,
            metrics=metrics, pendulum_pos_history=pendulum_pos_history
        )

    def _get_obs(self, data: mjx.Data, action: jp.ndarray) -> jp.ndarray:
        """Observes humanoid body position, velocities, angles, and hand contact."""
    
        position = data.qpos
        if self._exclude_current_positions_from_observation:
            position = position[2:]
    
        # Existing observations
        obs_list = [
            position,
            data.qvel,
            data.cinert[1:].ravel(),
            data.cvel[1:].ravel(),
            data.qfrc_actuator,
        ]
    
        # Add touch observations
        touch_obs = self.get_arm_touch_obs(data)
        obs_list.append(touch_obs)
    
        return jp.concatenate(obs_list)


# Register environment
envs.register_environment("humanoid", Humanoid)

# %%  instantiate the environment

env_name = 'humanoid'
env = envs.get_environment(env_name)

# define the jit reset/step functions
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

# %%  initialize the state

state = jit_reset(jax.random.PRNGKey(0))
rollout = [state.pipeline_state]

# grab a trajectory
for i in range(10):
  ctrl = -0.1 * jp.ones(env.sys.nu)
  state = jit_step(state, ctrl)
  rollout.append(state.pipeline_state)

#media.show_video(env.render(rollout, camera='side'), fps=1.0 / env.dt)

# Render frames from the environment
frames = env.render(rollout, camera='side')

# Save as an MP4 file
video_path = "rollout.mp4"
media.write_video(video_path, frames, fps=1.0 / env.dt)

# %% Train Humanoid Policy

train_fn = functools.partial(
    ppo.train, num_timesteps=30_000_000, num_evals=5, reward_scaling=0.1,
    episode_length=1000, normalize_observations=True, action_repeat=1,
    unroll_length=10, 
    num_minibatches=24, 
    num_updates_per_batch=8,
    discounting=0.97, 
    learning_rate=3e-4, 
    entropy_cost=1e-3, 
    num_envs=3072,
    batch_size=512, 
    seed=0)

x_data = []
y_data = []
ydataerr = []
times = [datetime.now()]

max_y, min_y = 13000, 0
def progress(num_steps, metrics):
  times.append(datetime.now())
  x_data.append(num_steps)
  y_data.append(metrics['eval/episode_reward'])
  ydataerr.append(metrics['eval/episode_reward_std'])

  plt.xlim([0, train_fn.keywords['num_timesteps'] * 1.25])
  plt.ylim([min_y, max_y])

  plt.xlabel('# environment steps')
  plt.ylabel('reward per episode')
  plt.title(f'y={y_data[-1]:.3f}')

  plt.errorbar(
      x_data, y_data, yerr=ydataerr)
  plt.show()

make_inference_fn, params, _= train_fn(environment=env, progress_fn=progress)

print(f'time to jit: {times[1] - times[0]}')
print(f'time to train: {times[-1] - times[1]}')

# %% Save Model

model_path = '/home/ajay/Python_Projects/mujoco-mjx/saved_models/mjx_brax_policy'
model.save_params(model_path, params)

# %% Load Model and Define Inference Function

params = model.load_params(model_path)

inference_fn = make_inference_fn(params)
jit_inference_fn = jax.jit(inference_fn)

# %% Visualize Policy

eval_env = envs.get_environment(env_name)

jit_reset = jax.jit(eval_env.reset)
jit_step = jax.jit(eval_env.step)

# initialize the state
rng = jax.random.PRNGKey(0)
state = jit_reset(rng)
rollout = [state.pipeline_state]

# grab a trajectory
n_steps = 500
render_every = 2

for i in range(n_steps):
  act_rng, rng = jax.random.split(rng)
  ctrl, _ = jit_inference_fn(state.obs, act_rng)
  state = jit_step(state, ctrl)
  rollout.append(state.pipeline_state)

  if state.done:
    break

#media.show_video(env.render(rollout[::render_every], camera='side'), fps=1.0 / env.dt / render_every)

# %% Visualize

import mediapy as media

# Generate frames
frames = env.render(rollout[::render_every], camera='side')
# Define video file path
video_path = "trainned_PPO_policy.mp4"
# Save frames as an MP4 video
media.write_video(video_path, frames, fps=1.0 / env.dt / render_every)
print(f"Video saved as: {video_path}")
