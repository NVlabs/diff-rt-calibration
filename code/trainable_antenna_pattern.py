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
from tensorflow.keras.layers import Layer
from sionna.constants import PI
from sionna.rt import polarization_model_1, polarization_model_2
from sionna.rt.utils import r_hat, normalize

class TrainableAntennaPattern(Layer):
    """Trainable antenna pattern

    This antenna pattern consists of a mixture of spherical Gaussians
    with trainable mean directions and concentration parameters.

    Parameters
    ----------
    num_mixtures : int
        Number of mixtures

    slant_angle : float
        Slant angle of the polarization model.
        Defaults  to 0.

    polarization_model : 1 | 2
        The polarization model from 3GPP TR 38.901 to be used.
        Defaults to 2.

    dtype : tf.complex64 | tf.complex128
        Datatype to be used.
        Defaults to tf.complex64

    Input
    -----
    theta : [array_like], tf.float
        Zenith angle

    phi = : [array_like], tf.float
        Azimuth angle

    Output
    ------
    c_theta : [array_like], tf.complex
        Zenith pattern

    c_phi : [array_like], tf.complex
        Azimuth pattern
    """
    def __init__(self,
                 num_mixtures,
                 slant_angle=0.0,
                 polarization_model=2,
                 dtype=tf.complex64):
        super(TrainableAntennaPattern, self).__init__()

        self._num_mixtures = num_mixtures
        self._polarization_model = polarization_model
        self._dtype = dtype
        self._rdtype = dtype.real_dtype
        self._slant_angle = tf.cast(slant_angle, self._rdtype)

    def build(self, input_shape):
        self._mu = tf.Variable(tf.initializers.GlorotUniform()(shape=(1, self._num_mixtures, 3)), dtype=self._rdtype)
        self._lambdas = tf.Variable(tf.initializers.GlorotUniform()(shape=(1, self._num_mixtures)), dtype=self._rdtype)
        self._weights = tf.Variable(tf.initializers.GlorotUniform()(shape=(1, self._num_mixtures)), dtype=self._rdtype)
        self._e_rad = tf.Variable((1), dtype=self._rdtype)

    def call(self, theta, phi):
        # Compute direction vectors
        theta = tf.cast(theta, self._rdtype)
        phi = tf.cast(phi, self._rdtype)
        v = tf.expand_dims(r_hat(theta, phi), -2)

        # Compute mean vectors and lambdas
        mu, _ = normalize(self._mu)
        lambdas = tf.abs(self._lambdas)

        # Compute scaling factor
        a = lambdas/tf.cast(2*PI, self._rdtype)/(1-tf.exp(-tf.cast(2, self._rdtype)*lambdas))

        # Compute PDFs
        gains = tf.cast(4*PI, self._rdtype) * a * tf.exp(lambdas*(tf.reduce_sum(v*mu, axis=-1) - tf.cast(1, self._rdtype)))

        # Compute weighted sum
        weights = tf.nn.softmax(self._weights)
        gain = tf.reduce_sum(gains*weights, axis=-1)

        # Add radiation efficieny
        gain *= self._e_rad

        # Compute antenna pattern from gain
        c = tf.complex(tf.sqrt(gain), tf.zeros_like(gain))
        if self._polarization_model==1:
            return polarization_model_1(c, theta, phi, self._slant_angle)
        else:
            return polarization_model_2(c, self._slant_angle)

