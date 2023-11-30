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
from sionna import PI, DIELECTRIC_PERMITTIVITY_VACUUM
from sionna.utils import expand_to_rank, flatten_last_dims
import pickle

from tensorflow.keras.layers import Layer, Dense


class NeuralMaterials(Layer):
    """Neural Network-based trainable materials

    This layer implements an MLP that computes the material properties
    from positions in the scene.

    Parameters
    ----------
    scene : Scene
        Instance of the loaded scene

    pos_encoding_size : int
        Size of the positional encoding.
        Defaults to 10.

    learn_scattering : bool
        If set to `False`, then zero-valued tensors are returned for the
        scattering and XPD coefficients.
        Defaults to `True`.

    Input
    -----
    objects [batch_dims], tf.int
        Integers uniquely identifying the intersected objects
        Not used.

    points : [batch_dims, 3], tf.float
        Positions of the intersection points

    Output
    ------
    complex_relative_permittivity : [num_objects], tf.float32
        Complex-valued relative permittivity

    scattering_coefficient : [num_objects], tf.float32
        Scattering coefficient.
        Returns a zero-valued tensor if `learn_scattering` is set to `False`.

    xpd_coefficient : [num_objects], tf.float32
        XPD coefficient.
        Returns a zero-valued tensor if `learn_scattering` is set to `False`.
    """

    def __init__(self, scene, pos_encoding_size=10, learn_scattering=True):
        super(NeuralMaterials, self).__init__()

        self.frequency = scene.frequency
        self.num_hidden_layers = 4
        self.num_units_per_hidden_layer = 128
        self.pos_encoding_size = pos_encoding_size
        self.learn_scattering = learn_scattering

        # For scaling to the bounding box
        x_min = scene.mi_scene.bbox().min.x
        y_min = scene.mi_scene.bbox().min.y
        z_min = scene.mi_scene.bbox().min.z

        x_max = scene.mi_scene.bbox().max.x
        y_max = scene.mi_scene.bbox().max.y
        z_max = scene.mi_scene.bbox().max.z

        self.center = tf.constant([0.5*(x_max + x_min),
                                   0.5*(y_max + y_min),
                                   0.5*(z_max + z_min)], tf.float32)

        scale = tf.constant([0.5*(x_max - x_min),
                             0.5*(y_max - y_min),
                             0.5*(z_max - z_min)], tf.float32)
        self.scale = tf.reduce_max(scale)

    def to_unit_bbox(self, x):
        """
        Center and scales `x` to the unit cube according to the scene
        bounding box
        """

        center_ = expand_to_rank(self.center, tf.rank(x), 0)
        scale_ = self.scale
        x_u = (x-center_)/scale_
        return x_u

    def pos_enc(self, x):
        """
        Positional encoding from:
        https://dl.acm.org/doi/pdf/10.1145/3503250
        """
        # x : [..., 3, 1], tf.float
        x = tf.expand_dims(x, axis=-1)

        # Tile x to fit the size of the positional encoding

        # Exponents
        # [pos_encoding_size]
        indices = tf.range(self.pos_encoding_size, dtype=tf.float32)
        # [..., 1, pos_encoding_size]
        indices = expand_to_rank(indices, tf.rank(x), 0)

        # Compute positional encoding
        # [..., 3, pos_encoding_size]
        enc = tf.pow(2., indices)*PI*x
        enc_cos = tf.math.cos(enc)
        enc_sin = tf.math.sin(enc)

        # [..., 3, 2*pos_encoding_size]
        enc = tf.concat([enc_cos, enc_sin], axis=-1)

        # Flatten feature dim
        # [..., 3*2*pos_encoding_size]
        enc = flatten_last_dims(enc, 2)

        return enc

    def complex_relative_permittivity(self, eta_prime, sigma):
        r"""
        Computes the complex relative permittivity from the relative permittivity
        and the conductivity
        """

        epsilon_0 = DIELECTRIC_PERMITTIVITY_VACUUM
        frequency = self.frequency
        omega = tf.cast(2.*PI*frequency, tf.float32)

        return tf.complex(eta_prime,
                          -tf.math.divide_no_nan(sigma, epsilon_0*omega))

    def build(self, input_shape):

        # Build the neural network
        layers = []

        for _ in range(self.num_hidden_layers):
            layers.append(Dense(self.num_units_per_hidden_layer, 'relu'))
        # Output layer has no activation and two outputs:
        # - Real relative permittivity
        # - Conductivity
        # - Scattering coefficient
        # - XPD
        layers.append(Dense(4, None))

        self.layers = layers

    def get_mat_props(self, pos):
        r"""
        Computes the material properties from the position
        """

        # pos : [..., 3], tf.float

        # Fit to unit cube
        pos = self.to_unit_bbox(pos)

        # Encode position
        # [..., 3*2*pos_encoding_size]
        enc_pos = self.pos_enc(pos)

        # MLP
        mat_prop = enc_pos
        for layer in self.layers:
            mat_prop = layer(mat_prop)
        # mat_prop : [..., 4]

        # [...]
        eta_prime = mat_prop[...,0] # Real relative_permittivity
        sigma = mat_prop[...,1] # Conductivity
        s = mat_prop[...,2] # Scattering coefficient
        xpd = mat_prop[...,3] # XPD

        return eta_prime, sigma, s, xpd

    def call(self, objects, points):

        # objects : [...], int
        # points : [..., 3], float

        # [...]
        eta_prime, sigma, s, xpd = self.get_mat_props(points)

        eta_prime = tf.exp(tf.clip_by_value(eta_prime, tf.math.log(1e-3), tf.math.log(200.))) + 1.
        sigma = tf.exp(tf.clip_by_value(sigma, tf.math.log(1e-3), tf.math.log(1e6)))
        s = tf.clip_by_value(tf.math.sigmoid(s), 1e-3, 1-1e-3)
        xpd = tf.clip_by_value(tf.math.sigmoid(xpd), 1e-3, 1-1e-3)

        # [...]
        eta = self.complex_relative_permittivity(eta_prime, sigma)

        if not self.learn_scattering:
            s = tf.zeros_like(s)
            xpd = tf.zeros_like(xpd)

        return eta, s, xpd
