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
from sionna.utils import expand_to_rank
from sionna.rt.utils import rot_mat_from_unit_vecs, dot, normalize

class TrainableScatteringPattern(Layer):
    """Trainable scattering pattern

    This scattering pattern consists of a mixture of a diffuse, directive,
    and backscattering componen. The last two have trainable concentration
    parameters.

    Parameters
    ----------
    dtype : tf.complex64 | tf.complex128
        Datatype to be used.
        Defaults to tf.complex64

    Input
    -----
    k_i : [batch_dims, 3], tf.float
        Incoming directions

    k_s : [batch_dims, 3], tf.float
        Outgoing directions

    n : [batch_dims, 3], tf.float
        Surface normals

    Output
    ------
    f_s : [batch_dims], tf.float
        Scattering pattern
    """
    def __init__(self, dtype=tf.complex64):
        super(TrainableScatteringPattern, self).__init__()
        self._dtype = dtype
        self._rdtype = dtype.real_dtype

    def build(self, input_shape):
        self._weights = tf.Variable([[1/3, 1/3, 1/3]], dtype=self._rdtype)
        self._lambda_r = tf.Variable([1], dtype=self._rdtype)
        self._lambda_i = tf.Variable([1], dtype=self._rdtype)

    @staticmethod
    def a_u(lambda_):
        return 2*PI/lambda_*(1-tf.exp(-lambda_))

    @staticmethod
    def a_b(lambda_):
        return 2*PI/lambda_*tf.exp(-2*lambda_)*(tf.exp(lambda_)-1)

    @staticmethod
    def t(lambda_):
        return tf.sqrt(lambda_+1e-12) * (1.6988*lambda_**2 + 10.8438*lambda_) / (lambda_**2 + 6.2201*lambda_ + 10.2415)

    @staticmethod
    def a_h(lambda_, beta):
        a = tf.exp(TrainableScatteringPattern.t(lambda_))
        b = tf.exp(TrainableScatteringPattern.t(lambda_)*tf.cos(beta))
        s = (a*b-1) / ((a-1)*(b+1))
        return TrainableScatteringPattern.a_u(lambda_)*s + TrainableScatteringPattern.a_b(lambda_)*(1-s)

    def call(self, object_id, points, k_i, k_s, n_hat):
        # Compute rotation matrix to bring all vectors to LCS
        z_hat = tf.constant([0, 0, 1], tf.float32)
        z_hat = tf.broadcast_to(z_hat, n_hat.shape)
        rot = rot_mat_from_unit_vecs(n_hat, z_hat)

        # Represent vectors in LCS
        n_hat = z_hat
        k_i = tf.linalg.matvec(rot, k_i)
        k_s = tf.linalg.matvec(rot, k_s)
                
        # Compute specular reflection
        k_r = k_i - 2 * dot(n_hat, k_i, keepdim=True) * n_hat

        # Lambertian pattern
        cos_theta_s = dot(k_s, n_hat, clip=True)
        pattern_l = cos_theta_s/tf.cast(PI, k_i.dtype)
        pattern_l = tf.where(pattern_l<0., tf.cast(0, self._rdtype), pattern_l)

        # Lobe in specular direction
        lambda_r = tf.squeeze(tf.clip_by_value(tf.abs(self._lambda_r), 1e-3, 30))
        mu_r = k_r
        z = tf.acos(tf.clip_by_value(mu_r[...,2], -1     , 1))
        q = tf.acos(tf.clip_by_value(mu_r[...,2], -1+1e-6, 1-1e-6))
        diff = tf.stop_gradient(z-q)
        beta_r = q + diff  
        dot_mu_r_k_s = dot(mu_r, k_s, clip=True)
        pattern_r = tf.exp(lambda_r*(dot_mu_r_k_s-1)) / self.a_h(lambda_r, beta_r)

        # Compute lobe in incoming direction
        lambda_i = tf.squeeze(tf.clip_by_value(tf.abs(self._lambda_i), 1e-3, 30))
        mu_i = -k_i
        z = tf.acos(tf.clip_by_value(mu_i[...,2], -1     , 1))
        q = tf.acos(tf.clip_by_value(mu_i[...,2], -1+1e-6, 1-1e-6))
        diff = tf.stop_gradient(z-q)
        beta_i = q + diff  
        dot_mu_i_k_s = tf.clip_by_value(dot(mu_i, k_s), -1, 1)
        pattern_i = tf.exp(lambda_i*(dot_mu_i_k_s-1)) / self.a_h(lambda_i, beta_i)

        # Compute weighted sum
        patterns = tf.stack([pattern_l, pattern_r, pattern_i], axis=0)
        weights, _ = normalize(tf.abs(self._weights))
        weights = expand_to_rank(weights[0]**2, tf.rank(patterns), axis=-1)
        pattern = tf.reduce_sum(patterns*weights, axis=0)

        return pattern
