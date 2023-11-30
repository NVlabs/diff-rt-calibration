# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Embedding
from sionna import PI, DIELECTRIC_PERMITTIVITY_VACUUM

class TrainableMaterials(Layer):
    """Trainable materials

    This layer computes overparametrized trainable radio material properties.

    Parameters
    ----------
    scene : Scene
        Instance of the loaded scene

    num_objects : int
        Number of objects in the scene

    embedding_size : int
        Size of the embeddings used to represent scalar parameters

    learn_scattering : bool
        If set to `False`, then zero-valued tensors are returned for the
        scattering and XPD coefficients.
        Defaults to `True`.

    Input
    -----
    None

    Output
    ------
    relative_permittivity : [num_objects], tf.float32
        Relative permittivity

    conductivity : [num_objects], tf.float32
        Conductivity

    scattering_coefficient : [num_objects], tf.float32
        Scattering coefficient.
        Returns a zero-valued tensor if `learn_scattering` is set to `False`.

    xpd_coefficient : [num_objects], tf.float32
        XPD coefficient.
        Returns a zero-valued tensor if `learn_scattering` is set to `False`.
    """
    def __init__(self, scene, num_objects, embedding_size=10, learn_scattering=True):
        super(TrainableMaterials, self).__init__()
        self.frequency = scene.frequency
        self.num_objects = num_objects
        self.embedding_size = embedding_size
        self.learn_scattering = learn_scattering

    def complex_relative_permittivity(self, eta_prime, sigma):
        r"""
        Computes the complex relative permittivity
        """
        epsilon_0 = DIELECTRIC_PERMITTIVITY_VACUUM
        frequency = self.frequency
        omega = tf.cast(2.*PI*frequency, tf.float32)

        return tf.complex(eta_prime,
                          -tf.math.divide_no_nan(sigma, epsilon_0*omega))

    def build(self, input_shape):
        self._v = tf.Variable(tf.initializers.GlorotUniform()(shape=(1, self.embedding_size)), name="mat-nn-v")
        self._w = tf.Variable(tf.initializers.GlorotUniform()(shape=(self.embedding_size, 4*self.num_objects)), name="mat-nn-w")


    def get_params(self):
        # Compute parameters in logarithmic domain
        epsilon_r, sigma, scattering_coefficient, xpd_coefficient = tf.split(tf.squeeze(tf.matmul(self._v, self._w)), 4, axis=0)

        epsilon_r = tf.squeeze(epsilon_r)
        sigma = tf.squeeze(sigma)
        scattering_coefficient = tf.squeeze(scattering_coefficient)
        xpd_coefficient = tf.squeeze(xpd_coefficient)

        # Clip to svae values for gradient computation and map to linear domain
        epsilon_r = tf.exp(tf.clip_by_value(epsilon_r, tf.math.log(1e-3), tf.math.log(200.))) + 1.
        sigma = tf.exp(tf.clip_by_value(sigma, tf.math.log(1e-3), tf.math.log(1e6)))
        scattering_coefficient = tf.clip_by_value(tf.math.sigmoid(scattering_coefficient), 1e-3, 1-1e-3)
        xpd_coefficient = tf.clip_by_value(tf.math.sigmoid(xpd_coefficient), 1e-3, 1-1e-3)

        if not self.learn_scattering:
            scattering_coefficient = tf.zeros_like(scattering_coefficient)
            xpd_coefficient = tf.zeros_like(xpd_coefficient)

        return epsilon_r, sigma, scattering_coefficient, xpd_coefficient

    def call(self, object_id, points):

        epsilon_r, sigma, scattering_coefficient, xpd_coefficient = self.get_params()

        # Compute complex relative permittivity
        eta = self.complex_relative_permittivity(epsilon_r, sigma)

        # Gather parameters corresponding to object_ids
        eta = tf.gather(eta, object_id)
        scattering_coefficient = tf.gather(scattering_coefficient, object_id)
        xpd_coefficient = tf.gather(xpd_coefficient, object_id)

        if not self.learn_scattering:
            scattering_coefficient = tf.zeros_like(scattering_coefficient)
            xpd_coefficient = tf.zeros_like(xpd_coefficient)

        return eta, scattering_coefficient, xpd_coefficient
