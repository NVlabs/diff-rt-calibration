# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import csv
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle
import json
import sys
from sionna.channel import subcarrier_frequencies, cir_to_ofdm_channel
from sionna.rt import normalize, rotation_matrix, theta_phi_from_unit_vec, \
                      PlanarArray, Transmitter, Receiver, load_scene, RadioMaterial, \
                      Paths, tr38901_pattern, r_hat
from sionna.constants import PI
from sionna.rt.solver_paths import PathsTmpData
from sionna.utils import expand_to_rank, log10

def change_coordinate_system(pos, center=None, rot_mat=None):
    """Transforms coordinates by applying an affine transformation

    Input
    ------
    pos : [batch_size, 3], float
        Coordinates

    center : [3], float or `None`
        Offset vector.
        Set to `None` for no offset.
        Defaults to `None`.

    rot_mat : [3,3], float or None
        Rotation matrix.
        Set to `None` to not apply a rotation.
        Defaults to `None`.

    Output
    -------
    pos : [batch_size, 3]
        Rotated and centered coordinates
    """
    if center is not None:
        pos -= center
    if rot_mat is not None:
        pos = tf.squeeze(tf.matmul(rot_mat, tf.reshape(pos, [3,1])))
    return pos

def get_coordinate_system():
    """Get points of interest as well as coordinate transformation parameters

    Output
    -------
    center : [3]
        Offset vector

    rot_mat : [3,3]
        Rotation matrix

    poi : dict
        Dictionary with points of interest and their coordinates
    """

    # Read and parse coordinates of points of interest (POIs)
    data_dict = {}
    with open('../data/coordinates.csv', mode ='r') as file:
        csv_file = csv.DictReader(file)
        for row in csv_file:
            data_dict[row['Name']] = {k: v for k, v in row.items() if k != 'Name'}

    poi = {}
    for val in data_dict:
        pos = data_dict[val]
        if not pos["South"]=="noLoS":
            poi[val] = tf.constant([float(pos["West"]),
                                    float(pos["South"]),
                                    float(pos["Height"])], tf.float32)

    # Add antenna array position
    poi["array_1"] = tf.constant([7.480775, -20.9824, 1.39335], tf.float32)
    poi["array_2"] = tf.constant([-6.390425, 24.440075, 1.4197], tf.float32)

    # Center coordinate system
    center = poi["AU"] # This is our new origin
    for p in poi:
        poi[p] = change_coordinate_system(poi[p], center)

    # Compute rotation matrix such that NWU has coordinates [0, 1, ~0]
    nwu_hat, _ = normalize(poi["NWU"])
    theta, phi = theta_phi_from_unit_vec(nwu_hat)
    rot_mat = tf.squeeze(rotation_matrix(tf.constant([[-phi.numpy()+PI/2, 0, 0]], tf.float32)))

    # Rotate all POIs to match the new coordinate system
    for p in poi:
        poi[p] = change_coordinate_system(poi[p], rot_mat=rot_mat)

    return center, rot_mat, poi

def load_dataset(name, calibrate=True, y_filter=None):
    """Loads the dataset and applies optionally calibration as well as offset
    corrections

    Input
    ------
    name : str
        Name of the dataset, e.g. "dichasus-dc01"

    calibrate : bool
        Determines if the dataset is calibrated. Note that the
        corresponding calibration file must be located in the folder
        "data/".

    y_filter : [2]
        The minimum and maximum y-coordinates of the measurement positions
        to be considered.

    Output
    -------
    dataset : A cached TFRecordDataset
        The data returns channel frequency responses of shape [batch_size, 64, 1024]
        as well as positions of shape [batch_size, 3].
    """

    raw_dataset = tf.data.TFRecordDataset(["../data/tfrecords/" + name + ".tfrecords"])

    center, rot_mat, _ = get_coordinate_system()

    feature_description = {
        "cfo": tf.io.FixedLenFeature([], tf.string, default_value = ''),
        "csi": tf.io.FixedLenFeature([], tf.string, default_value = ''),
        "gt-interp-age-tachy": tf.io.FixedLenFeature([], tf.float32, default_value = 0),
        "pos-tachy": tf.io.FixedLenFeature([], tf.string, default_value = ''),
        "snr": tf.io.FixedLenFeature([], tf.string, default_value = ''),
        "time": tf.io.FixedLenFeature([], tf.float32, default_value = 0),
    }

    def record_parse_function(proto):
        record = tf.io.parse_single_example(proto, feature_description)
        csi = tf.ensure_shape(tf.io.parse_tensor(record["csi"], out_type = tf.float32), (64, 1024, 2))
        csi = tf.signal.fftshift(csi, axes=1)
        csi = tf.complex(csi[...,0], csi[...,1])
        pos = tf.ensure_shape(tf.io.parse_tensor(record["pos-tachy"], out_type = tf.float64), (3))
        pos = tf.cast(pos, tf.float32)
        pos = change_coordinate_system(pos, center=center, rot_mat=rot_mat)
        return csi, pos

    def apply_calibration(csi, pos):
        """Apply STO and CPO calibration"""
        sto_offset = tf.tensordot(tf.constant(offsets["sto"]), 2 * np.pi * tf.range(tf.shape(csi)[1], dtype = np.float32) / tf.cast(tf.shape(csi)[1], np.float32), axes = 0)
        cpo_offset = tf.tensordot(tf.constant(offsets["cpo"]), tf.ones(tf.shape(csi)[1], dtype = np.float32), axes = 0)
        csi = tf.multiply(csi, tf.exp(tf.complex(0.0, sto_offset + cpo_offset)))
        return csi, pos

    dataset = raw_dataset.map(record_parse_function, num_parallel_calls = tf.data.experimental.AUTOTUNE)

    if calibrate:
        with open(f"../data/reftx-offsets-{name}.json", "r") as offsetfile:
            offsets = json.load(offsetfile)
        dataset = dataset.map(apply_calibration)

    if y_filter is not None:
        def position_filter(csi, pos):
            "Limit y-range to certain range to avoid position too close to the receiver"
            return pos[1] > y_filter[0] and  pos[1]<y_filter[1]
        dataset = dataset.filter(position_filter)

    return dataset.cache()

def detect_first_peak_2d_tensor(tensor):
    """Detects the index of the first peak in the last dimension of a 2D tensor

    Input
    ------
    tensor : [batch_size, n], tf.float
        Amplitudes of the taps

    Output
    -------
    peak_indices : [batch_size], tf.int32
        Index of the first significant peak
    """
    # Shift tensor one step to the right and left, padding with -inf to keep shape
    shifted_left = tf.pad(tensor[:, :-1], [[0, 0], [1, 0]], constant_values=float('-inf'))
    shifted_right = tf.pad(tensor[:, 1:], [[0, 0], [0, 1]], constant_values=float('-inf'))

    max_peak = tf.reduce_max(tensor, axis=-1, keepdims=True)

    # Find where tensor is greater than both its neighbors
    peaks = tf.math.logical_and(tensor > shifted_left, tensor > shifted_right)
    peaks = tf.cast(tf.math.logical_and(peaks, tensor > max_peak*0.1), tf.int16)

    # Get indices of peaks
    peak_indices = tf.argmax(peaks, output_type=tf.int32, axis=-1)
    return peak_indices

def freq2time(h, l_min=-8, l_max=80, peak_aligned=False, sampling_rate=50e6):
    """Transforms channel frequency response to time response

    Input
    ------
    h : [batch_size, num_subcarriers]
        Channel frequency response

    l_min : int
        Minimum tap index tp keep

    l_max : int
        Maximum tap index to keep

    peak_aligned: bool
        If `True`, the channel impulse response will be shifted in time such that
        the first significant peak appears at tap index 0.

    sampling_rate : float
        Sampling rate in Hz.
        Defaults to 50MHz.

    Output
    -------
    h_t : [batch_size, l_max-l_min+1]
        Channel impulse response

    tap_delays : [l_max-l_min+1] or [1, l_max-l_min+1]
        Tap delays in (s)
    """
    # h_t = tf.signal.fftshift(tf.signal.ifft(h), axes=-1)
    h_t = tf.signal.fftshift(tf.signal.ifft(tf.signal.ifftshift(h, axes=-1)), axes=-1)
    start = 512 + l_min
    stop = 512 + l_max
    h_t = h_t[...,start:stop]

    tap_ind = tf.range(l_min, l_max, dtype=tf.float32)

    if peak_aligned:
        peak_ind = tf.expand_dims(detect_first_peak_2d_tensor(tf.abs(h_t)), -1)
        tap_ind = tf.expand_dims(tap_ind, 0)
        tap_ind -= tf.cast(l_min, tf.float32) + tf.cast(peak_ind, tf.float32)

    # Compute tap delays in ns
    tap_delays = tap_ind / sampling_rate

    return h_t, tap_delays

def rms_delay_spread(h, tap_delays):
    """Computes RMS Delay spread

    Input
    ------
    h : [batch_size, num_taps]
        Channel impulse response

    tap_delays : [num_taps] or [1 or batch_size, num_taps]
        Tap delays

    Output
    -------
    tau_rms : [batch_size]
        Delay spread
    """
    a = tf.abs(h)**2
    tap_delays = expand_to_rank(tap_delays, tf.rank(a), axis=0)
    total_power = tf.reduce_sum(a, axis=-1, keepdims=True)
    time_weighted_power = tf.reduce_sum(tap_delays*a, axis=-1, keepdims=True)
    tau_bar =  time_weighted_power / total_power
    squared_delays = (tap_delays-tau_bar)**2
    tau_rms = tf.sqrt( tf.reduce_sum(squared_delays*a, axis=-1) / total_power[...,0] )
    return tau_rms

def init_scene(name, use_tx_array=True, tx_pattern="tr38901", rx_pattern="dipole"):
    """Loads a scene by name and configures transmit and receive arrays

    Input
    ------
    name : str
        Must correspond to the name of a valid scene in the `scenes` folder.

    use_tx_array : bool
        If set to `True`, then the transmitter array is configured to be
        a 4x8 array (as the one used for the measurements).
        Otherwise, a single-antenna array is set.
        Defaults to `True`.

    tx_pattern : str
        The antenna pattern to use at the transmitter.
        Defaults to "tr38901".

    rx_pattern : str
        The antenna pattern to use at the receiver.
        Defaults to "dipole".

    Output
    -------
    scene : sionna.rt.Scene
        Loaded scene
    """
    scene = load_scene(f"../scenes/{name}/{name}.xml")
    scene.frequency = 3.438e9
    scene.synthetic_array=False

    # Configure array
    if use_tx_array:
        num_rows = 4
        num_cols = 8
    else:
        num_rows = 1
        num_cols = 1

    scene.tx_array = PlanarArray(num_rows=num_rows,
                                 num_cols=num_cols,
                                 vertical_spacing=0.5,
                                 horizontal_spacing=0.5,
                                 pattern=tx_pattern,
                                 polarization="V")

    # This is the antenna used by the measurement robot
    scene.rx_array = PlanarArray(num_rows=1,
                                 num_cols=1,
                                 vertical_spacing=0.5,
                                 horizontal_spacing=0.5,
                                 pattern=rx_pattern,
                                 polarization="V")

    return scene

def select_transmitter(scene, tx_ind, ant_ind):

    tx_ind = int(tx_ind)
    ant_ind = int(ant_ind)

    # Get points of interest
    _, _, poi = get_coordinate_system()

    # Get center of antenna array
    center = poi[f"array_{tx_ind}"]

    # Get offset to center for the desired antenna
    array = PlanarArray(num_rows=4,
                        num_cols=8,
                        vertical_spacing=0.5,
                        horizontal_spacing=0.5,
                        polarization="V",
                        pattern="iso")

    # Compute final position
    position = center + array.positions[ant_ind]

    # Select correct orientation
    if tx_ind==1:
        orientation = [-PI/2, 0, 0]
    elif tx_ind==2:
        orientation = [PI/2, 0, 0]

    # Place transmitter
    tx = scene.get("tx")
    if tx is None:
        scene.add(Transmitter(name="tx",
                              position=position,
                              orientation=orientation))
    else:
        tx.position = position
        tx.orientation = orientation

def place_transmitter_arrays(scene, tx_indices):
    r"""
    Places transmitters arrays

    Input
    -------
    scene : sionna.rt.Scene
        Scene in which to place the transmitters

    tx_indices : `list` of `int`
        List of transmitters to place in `scene`
    """

    if isinstance(tx_indices, int):
        tx_indices = [tx_indices]

    # Get points of interest
    _, _, poi = get_coordinate_system()

    for tx_ind in tx_indices:

        # Get center of antenna array
        position = poi[f"array_{tx_ind}"]

        # Select correct orientation
        if tx_ind==1:
            orientation = [-PI/2, 0, 0]
        elif tx_ind==2:
            orientation = [PI/2, 0, 0]

        # Place transmitters
        scene.add(Transmitter(name=f"tx-{tx_ind}",
                              position=position,
                              orientation=orientation))

def total_power(h):
    """Computes the total power over the subcarriers

    Input
    ------
    h : [batch_size, num_subcarriers], tf.complex
        Channel frequency response

    Output
    -------
    : [batch_size], tf.float
        Total energy of each channel response
    """
    return tf.reduce_sum(tf.abs(h)**2, axis=-1)

def delay_spread_loss(h_rt, h_meas, return_ds=False):
    """Computes the RMS delay spread for all examples and the loss based on
       the symmetric mean absolute percentage error (SMAPE)

    Input
    ------
    h_rt : [batch_size, num_subcarriers], tf.complex
        Synthetic channel frequency response

    h_meas : [batch_size, num_subcarriers], tf.complex
        Ground truth channel frequency response

    return_ds : bool
        Return the delay spreads

    Output
    -------
    : tf.float
        Delay spread loss

    : tf.float
        Average delay spread of the synthetic channels.
        Only returned if ``return_ds`` is `True`.

    : tf.float
        Average delay spread of the measured channels.
        Only returned if ``return_ds`` is `True`.
    """
    h, t = freq2time(h_rt)
    ds_rt = rms_delay_spread(h, t*1e9)

    h, t = freq2time(h_meas)
    ds_meas = rms_delay_spread(h, t*1e9)

    loss = tf.reduce_mean(tf.math.divide_no_nan(tf.abs(ds_rt-ds_meas),
                                                tf.stop_gradient(ds_meas + ds_rt)))

    if return_ds:
        return loss, tf.reduce_mean(ds_rt), tf.reduce_mean(ds_meas)

    return loss


def power_loss(h_rt, h_meas, return_pow=False):
    """Computes the total power for all examples and the loss based on
       the symmetric mean absolute percentage error (SMAPE)

    Input
    ------
    h_rt : [batch_size, num_subcarriers], tf.complex
        Synthetic channel frequency response

    h_meas : [batch_size, num_subcarriers], tf.complex
        Ground truth channel frequency response

    return_po : bool
        Return the powers

    Output
    -------
    : tf.float
        Power loss

    : tf.float
        Average power of the synthetic channels.
        Only returned if ``return_ds`` is `True`.

    : tf.float
        Average power of the measured channels.
        Only returned if ``return_ds`` is `True`.
    """
    pow_rt = total_power(h_rt)
    pow_meas = total_power(h_meas)

    loss = tf.reduce_mean(tf.math.divide_no_nan(tf.abs(pow_rt-pow_meas),
                                                tf.stop_gradient(pow_meas + pow_rt)))

    if return_pow:
        return loss, tf.reduce_mean(pow_rt), tf.reduce_mean(pow_meas)

    return loss

def mse_power_scaling_factor(h_rt, h_meas):
    """Scaling factor alpha that minimizes:
        sum_i (pow_rt_i - alpha*pow_meas_i)^2

    Input
    ------
    h_rt : [batch_size, num_subcarriers], tf.complex
        Synthetic channel frequency response

    h_meas : [batch_size, num_subcarriers], tf.complex
        Ground truth channel frequency response

    Output
    -------
    : [batch_size], tf.float
        Scaling factor
    """
    pow_rt = tf.reduce_sum(tf.abs(h_rt)**2, axis=-1)
    pow_meas = tf.reduce_sum(tf.abs(h_meas)**2, axis=-1)
    scaling_factor = tf.reduce_sum(pow_rt*pow_meas) / tf.reduce_sum(pow_meas**2)
    return scaling_factor

def instantiate_receivers(scene, num_rx):
    """
    Removes all receivers from ``scene`` and instantiates ``num_rx`` receivers

    Input
    ------
    scene: sionna.rt.Scene
        Scene

    num_rx : int
        Number of receivers to instantiate
    """

    rx_names = scene.receivers.keys()
    for rx in rx_names:
        scene.remove(rx)

    for i in range(num_rx):
        name = f'rx-{i}'
        if scene.get(name) is None:
            scene.add(Receiver(name=f"rx-{i}", position=(0.,0.,0.)))

def set_receiver_positions(scene, rx_pos):
    """
    Sets positions of the receivers in ``scene``

    Input
    ------
    scene: sionna.rt.Scene
        Scene

    rx_pos : [num_rx, 3], tf.float
        Positions of the receivers
    """

    num_rx = rx_pos.shape[0]
    for i in range(num_rx):
        name = f'rx-{i}'
        scene.receivers[name].position = rx_pos[i]

def serialize_traced_paths(rx_pos, h_meas, traced_paths, squeeze_target_dim):
    """
    Serializes the traced paths

    Input
    ------
    rx_pos : [3], tf.float
        Position of the receiver

    h_meas : [1, num_tx*num_tx_ant=64, num_subcarriers=1024], tf.float
        Tensor of measurements

    traced_paths : list
        The traced paths as generated by `Scene.trace_paths()`

    squeeze_target_dim : bool
        If set to `True`, the target dimension is squeezed.
        This helps with creating batches.

    Output
    ------
    : str
        Serialized traced paths
    """

    # Map Paths objects to a single dictionary of tensor
    dict_list_ = [x.to_dict() for x in traced_paths]

    # Target axis index
    target_axis = {'a' : 0,
                   'tau' : 0,
                   'mask' : 0,
                   'objects' : 1,
                   'phi_r' : 0,
                   'phi_t' : 0,
                   'theta_r' : 0,
                   'theta_t' : 0,
                   'vertices' : 1,
                   'targets' : 0,
                   # Paths TMP data
                   'normals' : 1,
                   'k_i' : 1,
                   'k_r' : 1,
                   'total_distance' : 0,
                   'mat_t' : 0,
                   'k_tx' : 0,
                   'k_rx' : 0,
                   'scat_last_objects' : 0,
                   'scat_last_vertices' : 0,
                   'scat_last_k_i' : 0,
                   'scat_k_s' : 0,
                   'scat_last_normals' : 0,
                   'scat_src_2_last_int_dist' : 0,
                   'scat_2_target_dist' : 0}

    # Remove useless tensors and drop the target axis
    dict_list = []
    for d_ in dict_list_:
        d = {}
        for k in d_.keys():
            # Drop useless tensors
            if not k.startswith('scat_prefix_'):
                d.update({k : d_[k]})
                # Squeezes target dimension if requested
                if squeeze_target_dim and (k in target_axis):
                    d[k] = tf.squeeze(d[k], axis=target_axis[k])
        dict_list.append(d)

    # Add a prefix to indicate to which object each tensor belongs to
    all_tensors = {}
    all_tensors.update({'spec-' + k : dict_list[0][k] for k in dict_list[0]})
    all_tensors.update({'diff-' + k : dict_list[1][k] for k in dict_list[1]})
    all_tensors.update({'scat-' + k : dict_list[2][k] for k in dict_list[2]})
    all_tensors.update({'tmp-spec-' + k : dict_list[3][k] for k in dict_list[3]})
    all_tensors.update({'tmp-diff-' + k : dict_list[4][k] for k in dict_list[4]})
    all_tensors.update({'tmp-scat-' + k : dict_list[5][k] for k in dict_list[5]})

    # Add the receiver position
    all_tensors.update({'rx_pos' : rx_pos})

    # Add the channel measurement
    all_tensors.update({'h_meas' : h_meas})

    # Serialize the tensors to a string of bytes
    for k,v in all_tensors.items():
        all_tensors[k] =  tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(v).numpy()]))

    ex = tf.train.Example(features=tf.train.Features(feature=all_tensors))
    record_bytes = ex.SerializeToString()
    return record_bytes

def reshape_h_meas(h_meas_raw):
    """
    Reshapes ``h_meas_raw`` to separate the two transmitters

    Input
    ------
    h_meas_raw : [1, num_tx*num_tx_ant=64, num_subcarriers=1024], tf.float
        Tensor of measurements

    Output
    ------
     : [num_tx=2, num_tx_ant=32, num_subcarriers=1024], tf.float
        Reshaped tensor of measurements
    """

    # Indices of the antennas for each transmitter
    tx_1_ind = tf.constant([58, 57, 63, 34,
                            12,  7, 13,  9,
                            24, 45, 55, 36,
                            9, 18, 37, 32,
                            16, 42, 46, 22,
                             2, 52, 50, 62,
                            33, 28, 43, 39,
                            38, 21, 51, 10], tf.int32)

    tx_2_ind = tf.constant([60, 40, 44,  3,
                            54,  8, 53, 11,
                             6,  0, 61, 47,
                            17, 27, 59, 30,
                            49, 41, 48, 14,
                            20, 25, 35,  4,
                             5, 23,  1, 15,
                            19, 56, 26, 31], tf.int32)

    # Extract the antennas for each transmitter
    # [num_tx = 1, num_tx_and = 32, nu_subcarriers = 1024]
    h_meas_tx_1 = tf.gather(h_meas_raw, tx_1_ind, axis=1)
    h_meas_tx_2 = tf.gather(h_meas_raw, tx_2_ind, axis=1)

    # Concatenate into one tensor
    # [num_tx = 2, num_tx_and = 32, nu_subcarriers = 1024]
    h_meas = tf.concat([h_meas_tx_1, h_meas_tx_2], axis=0)

    return h_meas

def deserialize_paths_as_tensor_dicts(serialized_item):
    """
    Deserializes examples of a dataset of traced paths

    Input
    -----
    serialized_item : str
        A stream of bytes

    Output
    -------
    rx_pos : [3], tf.float
        Position of the receiver

    h_meas : [num_tx*num_tx_ant=64, num_subcarriers=1024], tf.complex
        Measured CSI

    spec_data : dict
        Dictionary of LoS and specular traced paths

    diff_data : dict
        Dictionary of diffracted traced paths

    scat_data : dict
        Dictionary of scattered traced paths

    tmp_spec_data : dict
        Dictionary of additional LoS and specular traced paths data

    tmp_diff_data : dict
        Dictionary of additional diffracted traced paths data

    tmp_scat_data : dict
        Dictionary of additional scattered traced paths data
    """

    # Fields names and types
    paths_fields_dtypes = {'a' : tf.complex64,
                           'mask' : tf.bool,
                           'normalize_delays' : tf.bool,
                           'objects' : tf.int32,
                           'phi_r' : tf.float32,
                           'phi_t' : tf.float32,
                           'reverse_direction' : tf.bool,
                           'sources' : tf.float32,
                           'targets' : tf.float32,
                           'tau' : tf.float32,
                           'theta_r' : tf.float32,
                           'theta_t' : tf.float32,
                           'types' : tf.int32,
                           'vertices' : tf.float32}
    tmp_paths_fields_dtypes = {'k_i' : tf.float32,
                               'k_r' : tf.float32,
                               'k_rx' : tf.float32,
                               'k_tx' : tf.float32,
                               'normals' : tf.float32,
                               'scat_2_target_dist' : tf.float32,
                               'scat_k_s' : tf.float32,
                               'scat_last_k_i' : tf.float32,
                               'scat_last_normals' : tf.float32,
                               'scat_last_objects' : tf.int32,
                               'scat_last_vertices' : tf.float32,
                               'scat_src_2_last_int_dist' : tf.float32,
                               'sources' :tf.float32,
                               'targets' : tf.float32,
                               'total_distance' : tf.float32,
                               'num_samples' : tf.int32,
                               'scat_keep_prob' : tf.float32}
    members_dtypes = {}
    members_dtypes.update({'spec-' + k : paths_fields_dtypes[k] for k in paths_fields_dtypes})
    members_dtypes.update({'diff-' + k : paths_fields_dtypes[k] for k in paths_fields_dtypes})
    members_dtypes.update({'scat-' + k : paths_fields_dtypes[k] for k in paths_fields_dtypes})
    members_dtypes.update({'tmp-spec-' + k : tmp_paths_fields_dtypes[k] for k in tmp_paths_fields_dtypes})
    members_dtypes.update({'tmp-diff-' + k : tmp_paths_fields_dtypes[k] for k in tmp_paths_fields_dtypes})
    members_dtypes.update({'tmp-scat-' + k : tmp_paths_fields_dtypes[k] for k in tmp_paths_fields_dtypes})
    members_dtypes.update({'tmp-scat-' + k : tmp_paths_fields_dtypes[k] for k in tmp_paths_fields_dtypes})

    # Add the receiver position
    members_dtypes.update({'rx_pos' : tf.float32})

    # Add channel measurement
    members_dtypes.update({'h_meas' : tf.complex64})

    # Build dict of tensors
    # Deserializes the byte stream corresponding to each tensor
    features = {k : tf.io.FixedLenFeature([], tf.string, default_value = '') for k in members_dtypes}
    record = tf.io.parse_single_example(serialized_item, features)
    members_data = {k : tf.io.parse_tensor(record[k], out_type = members_dtypes[k]) for k in members_dtypes}

    # Builds the paths objects
    spec_data = {k[len('spec-'):] : members_data[k] for k in members_data if k.startswith('spec-')}
    diff_data = {k[len('diff-'):] : members_data[k] for k in members_data if k.startswith('diff-')}
    scat_data = {k[len('scat-'):] : members_data[k] for k in members_data if k.startswith('scat-')}
    tmp_spec_data = {k[len('tmp-spec-'):] : members_data[k] for k in members_data if k.startswith('tmp-spec-')}
    tmp_diff_data = {k[len('tmp-diff-'):] : members_data[k] for k in members_data if k.startswith('tmp-diff-')}
    tmp_scat_data = {k[len('tmp-scat-'):] : members_data[k] for k in members_data if k.startswith('tmp-scat-')}

    # Retrieve receiver position
    rx_pos = members_data['rx_pos']

    # Retrieve channel measurement
    h_meas = members_data['h_meas']

    return rx_pos, h_meas, spec_data, diff_data, scat_data, tmp_spec_data, tmp_diff_data, tmp_scat_data

def tensor_dicts_to_traced_paths(scene, tensor_dicts):
    """
    Creates Sionna `Paths` and `PathsTmpData` objects from dictionaries

    Input
    ------
    scene : Sionna.rt.Scene
        Scene

    tensor_dicts : `list` of `dict`
        List of dictionaries, as retrieved when iterating over a dataset
        of traced paths and using ``deserialize_paths_as_tensor_dicts()``
        to retrieve the data.

    Output
    -------
    spec_paths : Paths
        Specular paths

    diff_paths : Paths
        Diffracted paths

    scat_paths : Paths
        Scattered paths

    spec_paths_tmp : PathsTmpData
        Additional data required to compute the EM fields of the specular
        paths

    diff_paths_tmp : PathsTmpData
        Additional data required to compute the EM fields of the diffracted
        paths

    scat_paths_tmp : PathsTmpData
        Additional data required to compute the EM fields of the scattered
        paths
    """

    sources = tensor_dicts[0]['sources']
    targets = tensor_dicts[0]['targets']

    spec_paths = Paths(sources, targets, scene)
    spec_paths.from_dict(tensor_dicts[0])

    diff_paths = Paths(sources, targets, scene)
    diff_paths.from_dict(tensor_dicts[1])

    scat_paths = Paths(sources, targets, scene)
    scat_paths.from_dict(tensor_dicts[2])

    spec_tmp_paths = PathsTmpData(sources, targets, scene.dtype)
    spec_tmp_paths.from_dict(tensor_dicts[3])

    diff_tmp_paths = PathsTmpData(sources, targets, scene.dtype)
    diff_tmp_paths.from_dict(tensor_dicts[4])

    scat_tmp_paths = PathsTmpData(sources, targets, scene.dtype)
    scat_tmp_paths.from_dict(tensor_dicts[5])

    return (spec_paths, diff_paths, scat_paths, spec_tmp_paths, diff_tmp_paths, scat_tmp_paths)

def pad_traced_paths(traced_paths, max_num_paths_spec, max_num_paths_diff, max_num_paths_scat):
    """
    Pads traced paths such that they all have the same maximum number of paths

    The paths added for padding are masked.

    Input
    ------
    traced_paths : `list`
        List of `Paths` and `PathsTmpData` objects, as returned by
        ``tensor_dicts_to_traced_paths()``

    max_num_paths_spec : int
        Maximum number of LoS and specular paths over the dataset.
        Padding will be added to reach that number.

    max_num_paths_diff : int
        Maximum number of diffracted paths over the dataset.
        Padding will be added to reach that number.

    max_num_paths_scat : int
        Maximum number of scattered paths over the dataset.
        Padding will be added to reach that number.

    Output
    -------
    spec_paths : Paths
        Specular paths

    diff_paths : Paths
        Diffracted paths

    scat_paths : Paths
        Scattered paths

    spec_paths_tmp : PathsTmpData
        Additional data required to compute the EM fields of the specular
        paths

    diff_paths_tmp : PathsTmpData
        Additional data required to compute the EM fields of the diffracted
        paths

    scat_paths_tmp : PathsTmpData
        Additional data required to compute the EM fields of the scattered
        paths
    """

    ### Function that pads a `Paths` object
    def _pad_paths(paths, max_num_paths):
        axis_to_pad = {'mask' : -1,
                       'objects' : -1,
                       'phi_r' : -1,
                       'phi_t' : -1,
                       'tau' : -1,
                       'theta_r' : -1,
                       'theta_t' : -1,
                       'vertices' : -2}

        paths_dicts = paths.to_dict()
        for k,a in axis_to_pad.items():
            t = paths_dicts[k]

            r = tf.rank(t).numpy()
            dim = r + a

            num_paths = t.shape[dim]
            if num_paths == 0:
                continue

            padding_size = max_num_paths - num_paths
            assert padding_size >= 0

            paddings = [[0,0]]*r
            paddings[dim] = [0,padding_size]

            if k == 'mask':
                t = tf.pad(t, paddings, constant_values=False) # Mask the paths added for padding
            elif k == 'tau':
                t = tf.pad(t, paddings, constant_values=-1.)
            elif k == 'objects':
                t = tf.pad(t, paddings, constant_values=-1)
            else:
                t = tf.pad(t, paddings, constant_values=0)
            paths_dicts[k] = t
        paths.from_dict(paths_dicts)

        return paths

    ### Function that pads a `PathsTmpData` object
    def _pad_tmp_paths(tmp_paths, max_num_paths):
        axis_to_pad = {'k_i' : -2,
                       'k_r' : -2,
                       'k_rx' : -2,
                       'k_tx' : -2,
                       'normals' : -2,
                       'scat_2_target_dist' : -1,
                       'scat_k_s' : -2,
                       'scat_last_k_i' : -2,
                       'scat_last_normals' : -2,
                       'scat_last_objects' : -1,
                       'scat_last_vertices' : -2,
                       'scat_src_2_last_int_dist' : -1,
                       'total_distance' : -1}

        paths_dicts = tmp_paths.to_dict()
        for k,a in axis_to_pad.items():
            t = paths_dicts[k]

            r = tf.rank(t).numpy()
            dim = r + a

            num_paths = t.shape[dim]
            if num_paths == 0:
                continue

            padding_size = max_num_paths - num_paths
            assert padding_size >= 0

            paddings = [[0,0]]*r
            paddings[dim] = [0,padding_size]

            t = tf.pad(t, paddings, constant_values=0)
            paths_dicts[k] = t
        tmp_paths.from_dict(paths_dicts)

        return tmp_paths

    # Tiling the paths
    spec_paths = _pad_paths(traced_paths[0], max_num_paths_spec)
    diff_paths = _pad_paths(traced_paths[1], max_num_paths_diff)
    scat_paths = _pad_paths(traced_paths[2], max_num_paths_scat)

    # Tiling the additional data paths
    tmp_spec_paths = _pad_tmp_paths(traced_paths[3], max_num_paths_spec)
    tmp_diff_paths = _pad_tmp_paths(traced_paths[4], max_num_paths_diff)
    tmp_scat_paths = _pad_tmp_paths(traced_paths[5], max_num_paths_scat)

    return spec_paths, diff_paths, scat_paths, tmp_spec_paths, tmp_diff_paths, tmp_scat_paths

def batchify(traced_paths_dicts):
    """
    Batchifies traced paths dictionaries

    This utility enables sampling batches of receivers positions from the dataset of traced paths.
    It arranges the receivers as targets, by concatenating and reshaping the tensors accordingly.

    Input
    ------
    tensor_dicts : `list` of `dict`
        List of dictionaries, as retrieved when iterating over a dataset
        of traced paths and using ``deserialize_paths_as_tensor_dicts()``
        to retrieve the data.

    Output
    -------
     : `list` of `dict`
        List of dictionaries
    """

    # Target axis index
    axis_to_swap = {'objects' : 1,
                    'vertices' : 1,
                    'k_i' : 1,
                    'k_r' : 1,
                    'normals' : 1}

    for d in traced_paths_dicts:
        # Swap axis 0 and targets if required
        # This is done because when batching a TF dataset, the batch dimension
        # is always axis 0, which might not be the target axis.
        for k in d.keys():
            if k not in axis_to_swap:
                continue
            v = d[k]
            a = axis_to_swap[k]
            perm = tf.range(tf.rank(v))
            perm = tf.tensor_scatter_nd_update(perm, [[0]], [a])
            perm = tf.tensor_scatter_nd_update(perm, [[a]], [0])
            d[k] = tf.transpose(v, perm)

        # Drop the batch dimension for sources, as these are the same
        # for all examples in the batch
        if 'sources' in d:
            d['sources'] = d['sources'][0]

        # Drop the batch dim for types
        if 'types' in d:
            d['types'] = d['types'][0]

    # De-batchify num_samples and scat_keep_prop
    traced_paths_dicts[-1]['num_samples'] = traced_paths_dicts[-1]['num_samples'][0]
    traced_paths_dicts[-1]['scat_keep_prob'] = traced_paths_dicts[-1]['scat_keep_prob'][0]
    traced_paths_dicts[-2]['num_samples'] = traced_paths_dicts[-2]['num_samples'][0]
    traced_paths_dicts[-2]['scat_keep_prob'] = traced_paths_dicts[-2]['scat_keep_prob'][0]
    traced_paths_dicts[-3]['num_samples'] = traced_paths_dicts[-3]['num_samples'][0]
    traced_paths_dicts[-3]['scat_keep_prob'] = traced_paths_dicts[-3]['scat_keep_prob'][0]

    return traced_paths_dicts

def split_dataset(dataset, dataset_size, training_set_size, validation_set_size,
                  test_set_size, shuffle_seed=42):
    r"""
    Splits the dataset by taking the first ``training_set_size`` elements to form
    the training set, and the last ``validation_set_size`` and ``test_set_size``
    elements to form the validation and test sets, respectively.

    The dataset is first shuffled using ``shuffle_seed`` as seed for the random
    number generator. Multiple calls to this function using the same seed leads
    to the same subsets to be created.

    Input
    ------
    dataset : tf.data.Dataset
        Dataset to split

    dataset_size : int
        Size of the dataset to split

    training_set_size : int
        Size of the training subset to create

    validation_set_size : int
        Size of the validation subset to create

    test_set_size : int
        Size of the test subset to create

    shuffle_seed : int
        Seed for shuffling the dataset before splitting.
        Defaults to 42.

    Output
    -------
    training_set : tf.data.Dataset
        Training subset

    validation_set : tf.data.Dataset
        Validation subset

    test_set : tf.data.Dataset
        Test subset
    """

    # Not reshuffle after each iteration to ensure that multiple calls to this
    # function leads to the same subsets to be created, assuming that the same
    # seed is used.
    tf.random.set_seed(42)
    shuffled_dataset = dataset.shuffle(256, seed=shuffle_seed,
                                       reshuffle_each_iteration=False)

    training_set = shuffled_dataset.take(training_set_size)
    test_validation_set = shuffled_dataset.skip(dataset_size-validation_set_size-test_set_size)
    validation_set = test_validation_set.take(validation_set_size)
    test_set = test_validation_set.skip(validation_set_size)

    return training_set, validation_set, test_set

def ale(p, p_ref):
    """
    Computes the absolute logarithmic error (ALE)

    Input
    ------
    p : [..., num_antenna]
        Batch of predictions

    p_ref : [..., num_antennas]
        Batch of reference values

    Output
    -------
    : [...]
        Absolute logarithmic error (ALE)
    """

    # Average over antennas
    # [...]
    p_avg = tf.reduce_mean(p, axis=-1)
    p_ref_avg = tf.reduce_mean(p_ref, axis=-1)

    # Linear to dB scale
    # [...]
    p_avg_db = 10.*log10(p_avg)
    p_ref_avg_db = 10.*log10(p_ref_avg)

    # ALE for each example and antenna
    # [...]
    ale_ = tf.abs(p_avg_db - p_ref_avg_db)

    return ale_

def relative_abs_error(p, p_ref):
    """
    Computes the relative absolute error

    Input
    ------
    p : [..., num_antenna]
        Batch of predictions

    p_ref : [..., num_antennas]
        Batch of reference values

    Output
    -------
    : [...]
        Absolute relative error
    """

    # Average over antennas
    # [...]
    p_avg = tf.reduce_mean(p, axis=-1)
    p_ref_avg = tf.reduce_mean(p_ref, axis=-1)

    # Absolute error for each example and antenna
    # [...]
    ae_ = tf.abs(p_avg - p_ref_avg)/p_ref_avg

    return ae_

def ds_ray_trace(scene, scaling_factor, params, test_set, batch_size,
                     test_set_size, num_subcarriers, bandwidth, scattering):
    """
    Computes the synthetic channel frequency response over the dataset
    ``test_set``.

    The measured CIRs are scaled by ``scaling_factor``.

    Input
    ------
    scaling_factor : float
        Scaling factor by which to scale the measurements

    params : `dict`
        Dataset parameters

    test_set : tf.data.Dataset
        Dataset

    batch_size : int
        Batch size to use for tracing

    test_set_size : int
        Size of the dataset ``test_set``

    num_subcarriers : int
        Number of subcarriers

    bandwidth : float
        Bandwidth [Hz]

    scattering : bool
        Enable/Disable scattering

    Output
    -------
    rx_pos : [test_set_size, 3], tf.float
        Receivers positions

    h_rt : [num_samples, num_tx, num_antenna, num_subcarriers], tf.complex
        Synthetic channel impulse responses generated by the ray tracer

    h_meas : [num_samples, num_tx, num_antenna, num_subcarriers], tf.complex
        Measured channel impulse responses
    """

    frequencies = subcarrier_frequencies(num_subcarriers, bandwidth/num_subcarriers)

    ###########################################
    # Function that runs a single evaluation
    # step
    ##########################################
    @tf.function
    def evaluation_step(rx_pos, h_meas, traced_paths):

        # Placer receiver
        set_receiver_positions(scene, rx_pos)

        # Build traced paths
        traced_paths = tensor_dicts_to_traced_paths(scene, traced_paths)

        paths = scene.compute_fields(*traced_paths,
                                     scat_random_phases=False,
                                     check_scene=False)

        a, tau = paths.cir(scattering=scattering)

        # Compute channel frequency response
        h_rt = cir_to_ofdm_channel(frequencies, a, tau)

        # Remove useless dimensions
        h_rt = tf.squeeze(h_rt, axis=[0,2,5])

        # Normalize h to make sure that power is independent of the number of subacrriers
        h_rt /= tf.complex(tf.sqrt(tf.cast(num_subcarriers, tf.float32)), 0.)

        # Scale measurements
        h_meas *= tf.complex(tf.sqrt(scaling_factor), 0.)

        return h_rt, h_meas

    ##########################################
    # Compute frequency domain CIRs over
    # the test set
    ##########################################

    h_rt = []
    h_meas = []
    rx_pos = []

    # Number of iterations
    num_test_iter = test_set_size // batch_size
    test_set_ter = iter(test_set.batch(batch_size))
    for next_item in range(num_test_iter):
        # Next set of traced paths
        next_item = next(test_set_ter, None)

        # Retreive the receiver position separately
        rx_pos_, h_meas_, traced_paths = next_item[0], next_item[1], next_item[2:]
        # Skip iteration if does not match the batch size
        if rx_pos_.shape[0] != batch_size:
            continue

        # Batchify
        traced_paths = batchify(traced_paths)

        # Evaluate
        eval_quantities = evaluation_step(rx_pos_, h_meas_, traced_paths)
        h_rt_, h_meas_ = eval_quantities
        h_rt.append(h_rt_)
        h_meas.append(h_meas_)
        rx_pos.append(rx_pos_)
    rx_pos = tf.concat(rx_pos, axis=0)
    h_rt = tf.concat(h_rt, axis=0)
    h_meas = tf.concat(h_meas, axis=0)

    return rx_pos, h_rt, h_meas

def cir2freq(a, tau):
    """Converts complex baseband channel impulse response to frequency response

    Input
    -----
    a : [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths, num_time_steps], tf.complex
        Path coefficients

    tau : [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths] , tf.real
        Path delays

    Output
    ------
    h : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, 1024], tf.complex
        Channel frequency response assuming a 50MHz 1024-subcarrier OFDM system
    """
    # Compute channel frequency response
    num_subcarriers = 1024
    bandwidth = 50e6
    frequencies = subcarrier_frequencies(num_subcarriers, bandwidth/num_subcarriers)
    h_rt = cir_to_ofdm_channel(frequencies, a, tau)

    # Normalize h to make sure that power is independent of the number of subacrriers
    h_rt /= tf.complex(tf.sqrt(tf.cast(num_subcarriers, tf.float32)), 0.)

    return h_rt

def save_model(layer, filename, scaling_factor):
    weights = layer.get_weights()
    weights.append(scaling_factor)
    with open(filename, 'wb') as f:
        pickle.dump(weights, f)

def load_model(layer, filename):
    with open(filename, 'rb') as f:
            weights = pickle.load(f)
    scaling_factor = weights[-1]
    weights = weights[:-1]
    layer.set_weights(weights)
    return scaling_factor

def plot_results_synthetic_data(train_dict, ground_truth, scene, save_figs=False):
    """Utility function to plot results for the calibration against a synthetic dataset"""
    fig_folder = "../results/synthetic/"

    ###
    ### Learned Materials
    ###
    iterations = np.array(train_dict["iterations"])
    for param in ground_truth["floor"]:
        plt.rcParams['font.size'] = 16
        plt.rcParams['font.family'] = 'serif'
        # activate to reproduce the exact figures from the paper
        # with latex backend
        plt.rcParams['text.usetex'] = False
        plt.figure(figsize=(7,4))
        legend = []
        for i, obj in enumerate(scene.objects):
            plt.plot(iterations, train_dict[obj][param], "-", c=f"C{i}")
        for i, obj in enumerate(scene.objects):
            plt.plot(iterations, np.ones_like(iterations)*ground_truth[obj][param], "--", c=f"C{i}")
            legend.append(obj.capitalize())
        plt.legend(legend, loc="upper right");
        plt.xlabel("Iteration")

        if param=="relative_permittivity":
            ylabel = r"Relative permittivity $\varepsilon_r$"
        elif param=="conductivity":
            ylabel=r"Conductivity $\sigma$"
        elif param=="scattering_coefficient":
            ylabel=r"Scattering coefficient $S$"
        elif param=="xpd_coefficient":
            ylabel=r"XPD coefficient $K_x$"
        plt.ylabel(ylabel, fontsize=14);
        plt.xlim([0, 3000])
        plt.tight_layout()
        if save_figs:
            plt.savefig(fig_folder + f"{param}.pdf")

    ###
    ### Antenna pattern
    ###
    # Vertical cut
    def comp_v(pattern):
        theta = np.linspace(0.0, PI, 1000)
        c_theta, c_phi = pattern(theta, np.zeros_like(theta))
        g = np.abs(c_theta)**2 + np.abs(c_phi)**2
        g = np.where(g==0, 1e-12, g)
        g_db = 10*np.log10(g)
        g_db_max = np.max(g_db)
        g_db_min = np.min(g_db)
        if g_db_min==g_db_max:
            g_db_min = -30
        else:
            g_db_min = np.maximum(-60., g_db_min)
        return theta, g_db, g_db_min, g_db_max

    # Horizontal cut
    def comp_h(pattern):
        phi = np.linspace(-PI, PI, 1000)
        c_theta, c_phi = pattern(PI/2*tf.ones_like(phi) ,
                                 tf.constant(phi, tf.float32))
        c_theta = c_theta.numpy()
        c_phi = c_phi.numpy()
        g = np.abs(c_theta)**2 + np.abs(c_phi)**2
        g = np.where(g==0, 1e-12, g)
        g_db = 10*np.log10(g)
        g_db_max = np.max(g_db)
        g_db_min = np.min(g_db)
        if g_db_min==g_db_max:
            g_db_min = -30
        else:
            g_db_min = np.maximum(-60., g_db_min)
        return phi, g_db, g_db_min, g_db_max

    fig_v = plt.figure(figsize=(4,4))
    theta, g_db, g_db_min, g_db_max = comp_v(tr38901_pattern)
    plt.polar(theta, g_db, "--")
    theta, g_db, g_db_min, g_db_max = comp_v(scene.tx_array.antenna.patterns[0])
    plt.polar(theta, g_db)
    fig_v.axes[0].set_rmin(g_db_min)
    fig_v.axes[0].set_rmax(g_db_max+3)
    fig_v.axes[0].set_theta_zero_location("N")
    fig_v.axes[0].set_theta_direction(-1)
    fig_v.axes[0].set_yticklabels([])
    plt.legend(["IEEE 38.901", "Learned"], loc='upper left')
    plt.tight_layout()
    if save_figs:
        plt.savefig(fig_folder + "antenna_pattern_v.pdf")

    fig_h = plt.figure(figsize=(4,4))
    phi, g_db, g_db_min, g_db_max = comp_h(tr38901_pattern)
    plt.polar(phi, g_db, "--")
    phi, g_db, g_db_min, g_db_max = comp_h(scene.tx_array.antenna.patterns[0])
    plt.polar(phi, g_db)
    fig_h.axes[0].set_rmin(g_db_min)
    fig_h.axes[0].set_rmax(g_db_max+3)
    fig_h.axes[0].set_theta_zero_location("E")
    fig_h.axes[0].set_yticklabels([])
    #plt.legend(["IEEE 38.901", "Learned"], loc='upper left')
    plt.tight_layout()
    if save_figs:
        plt.savefig(fig_folder + "antenna_pattern_h.pdf")

    ###
    ### Scattering pattern
    ###
    theta_i = PI/3
    k_i = -r_hat(theta_i, PI)
    n_hat = r_hat(0., 0.)

    learned_pattern = scene.scattering_pattern_callable
    theta_s = tf.cast(tf.linspace(0.0, PI/2, 100), dtype=tf.float32)
    phi_s = tf.broadcast_to(0., theta_s.shape)

    k_s = r_hat(theta_s, phi_s)
    k_i = tf.broadcast_to(k_i, k_s.shape)
    n_hat = tf.broadcast_to(n_hat, k_s.shape)

    fig_cut = plt.figure()
    plt.polar(theta_s, ground_truth["target_pattern"](k_i, k_s, n_hat), color='C0', linestyle="dashed")
    plt.polar(theta_s, learned_pattern(None, None, k_i, k_s, n_hat), color='C1')

    plt.polar(2*PI-theta_s, ground_truth["target_pattern"](k_i, r_hat(theta_s, phi_s-PI), n_hat), color='C0', linestyle="dashed")
    plt.polar(2*PI-theta_s, learned_pattern(None, None, k_i, r_hat(theta_s, phi_s-PI), n_hat), color='C1')

    plt.legend(["Backscattering Model", "Learned"], loc="best")

    ax = fig_cut.axes[0]
    xticks = [0, 30/180*np.pi, 60/180*np.pi, 90/180*np.pi, -30/180*np.pi, -60/180*np.pi, -90/180*np.pi]
    ax.set_xticks(xticks)
    ax.text(-theta_i-10*PI/180, ax.get_yticks()[-1]*2/3, r"$\hat{\mathbf{k}}_\mathrm{i}$", horizontalalignment='center')
    ax.text(theta_i+10*PI/180, ax.get_yticks()[-1]*2/3, r"$\hat{\mathbf{k}}_\mathrm{r}$", horizontalalignment='center')
    plt.quiver([0], [0], [np.sin(theta_i)], [np.cos(theta_i)], scale=1., color="grey",)
    plt.quiver([0], [0], [-np.sin(theta_i)], [np.cos(theta_i)], scale=1., color="grey",)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    plt.tight_layout()
    labels = [e.get_text() for e in ax.get_xticklabels()]
    labels[-2] = r"$\theta_i$"
    labels[2] = r"$\theta_r$"
    ax.set_xticklabels(labels);
    ax.set_yticklabels([]);

    if save_figs:
        plt.savefig(fig_folder + "scattering_pattern.pdf")
