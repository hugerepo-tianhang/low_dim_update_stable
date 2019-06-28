import numpy as np
import gym
import math
import tensorflow as tf


def ortho_init(scale=1.0):
    """
    Orthogonal initialization for the policy weights

    :param scale: (float) Scaling factor for the weights.
    :return: (function) an initialization function for the weights
    """

    # _ortho_init(shape, dtype, partition_info=None)
    def _ortho_init(shape, *_, **_kwargs):

        gaussian_noise = np.random.normal(0.0, 1.0, 1)
        u, _, v = np.linalg.svd(gaussian_noise, full_matrices=False)
        weights = u   # pick the one with the correct shape
        weights = weights.reshape(shape)
        return (1 * weights[:shape[0], :shape[1]]).astype(np.float32)

    return _ortho_init


weight = tf.get_variable("w", [1, 1], initializer=ortho_init(1))
