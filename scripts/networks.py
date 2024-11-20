"""Networks.
"""

from functools import partial
from typing import Sequence, Tuple, Any, Callable

from brax.training import distribution
from brax.training import networks
from brax.training import types
from brax.training.types import PRNGKey

import flax
from flax import linen
import jax
import jax.numpy as jp

ModuleDef = Any
ActivationFn = Callable[[jp.ndarray], jp.ndarray]
Initializer = Callable[..., Any]

class ManiSkillCNN(linen.Module):
  layer_sizes: Sequence[int]
  activation: ActivationFn = linen.relu
  kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
  activate_final: bool = False
  layer_norm: bool = False
  dtype: Any = jp.float32

  @linen.compact
  def __call__(self, data: jp.ndarray):
    conv = partial(linen.Conv, use_bias=True, dtype=self.dtype)
    hidden = data

    hidden = conv(features=16, kernel_size=(3, 3), name='conv1')(hidden)
    hidden = self.activation(hidden)
    hidden = linen.max_pool(hidden, window_shape=(2, 2), strides=(2, 2))

    hidden = conv(features=32, kernel_size=(3, 3), name='conv2')(hidden)
    hidden = self.activation(hidden)
    hidden = linen.max_pool(hidden, window_shape=(2, 2), strides=(2, 2))

    hidden = conv(features=64, kernel_size=(3, 3), name='conv3')(hidden)
    hidden = self.activation(hidden)
    hidden = linen.max_pool(hidden, window_shape=(2, 2), strides=(2, 2))

    hidden = conv(features=128, kernel_size=(3, 3), name='conv4')(hidden)
    hidden = self.activation(hidden)
    hidden = linen.max_pool(hidden, window_shape=(2, 2), strides=(2, 2))

    hidden = conv(features=128, kernel_size=(1, 1), name='conv5')(hidden)
    hidden = self.activation(hidden)

    hidden = jp.mean(hidden, axis=(-2, -3))

    for i, layer_size in enumerate(self.layer_sizes):
      hidden = linen.Dense(
        layer_size, kernel_init=self.kernel_init, name=f'dense_{i}')(hidden)
      if i != len(self.layer_sizes) - 1 or self.activate_final:
        hidden = self.activation(hidden)
        if self.layer_norm:
          hidden = linen.LayerNorm()(hidden)
    return hidden

class SimpleCNN(linen.Module):
  layer_sizes: Sequence[int]
  activation: ActivationFn = linen.relu
  kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
  activate_final: bool = False
  layer_norm: bool = False
  dtype: Any = jp.float32

  @linen.compact
  def __call__(self, data: jp.ndarray):
    conv = partial(linen.Conv, use_bias=False, dtype=self.dtype)
    hidden = data

    hidden = conv(features=32, kernel_size=(8, 8), strides=(4, 4), name='conv1')(hidden)
    hidden = self.activation(hidden)

    hidden = conv(features=64, kernel_size=(4, 4), strides=(2, 2), name='conv2')(hidden)
    hidden = self.activation(hidden)


    hidden = conv(features=64, kernel_size=(3, 3), strides=(1, 1), name='conv3')(hidden)
    hidden = self.activation(hidden)

    hidden = jp.mean(hidden, axis=(-2, -3))

    for i, layer_size in enumerate(self.layer_sizes):
      hidden = linen.Dense(
        layer_size, kernel_init=self.kernel_init, name=f'dense_{i}')(hidden)
      if i != len(self.layer_sizes) - 1 or self.activate_final:
        hidden = self.activation(hidden)
        if self.layer_norm:
          hidden = linen.LayerNorm()(hidden)
    return hidden


if __name__ == '__main__':
  rng = jax.random.PRNGKey(0)
  model = SimpleCNN(layer_sizes=[256, 256, 4], activation=linen.relu)
  params = model.init(rng, jp.ones((1, 64, 64, 4)))

  print(jax.tree_util.tree_map(lambda x: x.shape, params))
  x = jp.zeros((8, 64, 64, 4))
  ret = model.apply(params, x)

  print(ret.shape)
  assert(ret.shape == (8, 4))

  # tabulate_fn = linen.tabulate(model, rng)
  # print(tabulate_fn(x))
  print('SimpleCNN test passed')