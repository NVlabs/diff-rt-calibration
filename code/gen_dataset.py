# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


"""
Script to generate a dataset of traced paths
"""

###########################################
# Parse arguments
###########################################

import argparse

## Define the expected arguments and their default value
parser = argparse.ArgumentParser(description='Generate dataset of traced paths',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-gpu_num', type=int, help='Index of the GPU to use',
                    default=0)
#
parser.add_argument('-seed', type=int, help='Tensorflow seed',
                    default=1)
#
parser.add_argument('-meas_ds', type=str, help='Name of the dataset of measurements',
                    default="dichasus-dc01")
#
parser.add_argument('-scene_name', type=str, help='Sionna scene to use for ray tracing',
                    default="inue_simple")
#
parser.add_argument('-num_samples', type=int, help='Number of samples used for tracing',
                    default=int(4e6))
#
parser.add_argument('-max_depth', type=int, help='Maximum depth used for tracing',
                    default=5)
#
parser.add_argument('-los', help='Enables LoS when tracing', default=True, action='store_true')
parser.add_argument('-no-los', action='store_false', dest='los')
#
parser.add_argument('-reflection', help='Enables reflection when tracing', default=True, action='store_true')
parser.add_argument('-no-reflection', action='store_false', dest='reflection')
#
parser.add_argument('-diffraction', help='Enables diffraction when tracing', default=True, action='store_true')
parser.add_argument('-no-diffraction', action='store_false', dest='diffraction')
#
parser.add_argument('-edge_diffraction', help='Enables edge diffraction when tracing', default=False, action='store_true')
parser.add_argument('-no-edge_diffraction', action='store_false', dest='edge_diffraction')
#
parser.add_argument('-scattering', help='Enables scattering when tracing', default=True, action='store_true')
parser.add_argument('-no-scattering', action='store_false', dest='scattering')
#
parser.add_argument('-scat_keep_prob', type=float, help='Probability to keep a scattered paths when tracing',
                    default=0.001)
parser.add_argument('-traced_paths_dataset', type=str, help='(Required) Filename of the dataset of traced paths to create',
                    required=True)
parser.add_argument('-traced_paths_dataset_size', type=int, help='(Required) Size of the dataset of traced paths',
                    required=True)
parser.add_argument('-delete_raw_dataset', help='Deletes the raw (unpost-processed) dataset', default=True, action='store_true')
parser.add_argument('-keep_raw_dataset', action='store_false', dest='delete_raw_dataset')

## Parse arguments
args = parser.parse_args()
# GPU index to use
gpu_num = args.gpu_num
# Tensorflow seed
seed = args.seed
# Name of the dataset of measurments
meas_ds = args.meas_ds
# Sionna scene to use for ray tracing
scene_name = args.scene_name
# Number of samples used for tracing
num_samples = args.num_samples
# Maximum depth used for tracing
max_depth = args.max_depth
# Enables LoS when tracing
los = args.los
# Enables reflection when tracing
reflection = args.reflection
# Enables diffraction when tracing
diffraction = args.diffraction
# Enables edge diffraction when tracing
edge_diffraction = args.edge_diffraction
# Enables scattering when tracing
scattering = args.scattering
# Probability to keep a scattered paths when tracing
scat_keep_prob = args.scat_keep_prob
# Filename of the dataset of traced paths to create
traced_paths_dataset = args.traced_paths_dataset
# Size of the dataset of traced paths
# Set to -1 to match the datset of measurements
traced_paths_dataset_size = args.traced_paths_dataset_size
# Delete the raw dataset once post-processed?
delete_raw_dataset = args.delete_raw_dataset
# Folder where to save the dataset
traced_paths_dataset_folder = '../data/traced_paths'


###########################################
# Imports
###########################################

import os

os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
tf.get_logger().setLevel('ERROR')

# Set the seed
tf.random.set_seed(seed)

import sys
sys.path.append('../../..')
sys.path.append('../code')

# import sionna
from utils import *
import json


###########################################
# Setup the scene
###########################################

# Load the scene
scene = init_scene(scene_name, use_tx_array=True)

# Place the transmitters
place_transmitter_arrays(scene, [1,2])

# Instantiate the receivers
instantiate_receivers(scene, 1)


#########################################
# Generate a dataset of traced paths
#########################################

# Load the dataset of measurements.
dataset = load_dataset(meas_ds, calibrate=True, y_filter=[-15, 25])
dataset = dataset.batch(1).repeat()

# Build an iterator on the dataset
data_iter = iter(dataset)

traced_paths_raw_dataset_datafile = os.path.join(traced_paths_dataset_folder, traced_paths_dataset + '_raw.tfrecords')

# Iterate through all the positions in the dataset
# Length of the line printed to show progress
line_length = 50
# and trace the paths
# File writer to save the dataset
file_writer = tf.io.TFRecordWriter(traced_paths_raw_dataset_datafile)
# Keep track of the max_num_paths
max_num_paths_spec = -1
max_num_paths_diff = -1
max_num_paths_scat = -1
for it_num in range(traced_paths_dataset_size):

    # Retrieve the next item
    # `None` is returned if the iterator is exhausted
    next_item = next(data_iter, None)
    # Stop if exhausted
    if next_item is None:
        break

    # Retrieve the position
    h_meas_raw, rx_pos = next_item

    # Place the receiver
    set_receiver_positions(scene, rx_pos)

    # Trace the paths
    traced_paths = scene.trace_paths(num_samples=num_samples,
                                     max_depth=max_depth,
                                     los=los,
                                     reflection=reflection,
                                     diffraction=diffraction,
                                     edge_diffraction=edge_diffraction,
                                     scattering=scattering,
                                     scat_keep_prob=scat_keep_prob,
                                     check_scene=False)

    # Update max_num_paths
    num_paths_spec = traced_paths[0].objects.shape[-1]
    num_paths_diff = traced_paths[1].objects.shape[-1]
    num_paths_scat = traced_paths[2].objects.shape[-1]

    if num_paths_spec > max_num_paths_spec:
        max_num_paths_spec = num_paths_spec
    if num_paths_diff > max_num_paths_diff:
        max_num_paths_diff = num_paths_diff
    if num_paths_scat > max_num_paths_scat:
        max_num_paths_scat = num_paths_scat

    # Reshape the channel measurement
    h_meas = reshape_h_meas(h_meas_raw)

    # Serialize the traced paths
    record_bytes = serialize_traced_paths(rx_pos[0], h_meas, traced_paths, True)

    # Save the traced paths
    file_writer.write(record_bytes)

    # Print progress
    progress_message = f"\rProgress: {it_num+1}/{traced_paths_dataset_size}"
    progress_message = progress_message.ljust(line_length)
    print(progress_message, end="")

file_writer.close()
print("")
print("Raw dataset generated.")
print(f"Maximum number of paths:\n\tLoS + Specular: {max_num_paths_spec}\n\tDiffracted: {max_num_paths_diff}\n\tScattered: {max_num_paths_scat}")


#########################################
# Post-process the generated raw dataset
#########################################

print("Post-processing the raw dataset...")

raw_dataset = tf.data.TFRecordDataset([traced_paths_raw_dataset_datafile])
raw_dataset = raw_dataset.map(deserialize_paths_as_tensor_dicts)
raw_dataset_iter = iter(raw_dataset)

# Iterate through all the dataset and tile the paths to the same ``max_num_paths``
# File writer to save the dataset
traced_paths_dataset_datafile = os.path.join(traced_paths_dataset_folder, traced_paths_dataset + '.tfrecords')
file_writer = tf.io.TFRecordWriter(traced_paths_dataset_datafile)
for it_num in range(traced_paths_dataset_size):

    # Retrieve the next item
    # `None` is returned if the iterator is exhausted
    next_item = next(raw_dataset_iter, None)
    # Stop if exhausted
    if next_item is None:
        break

    # Retreive the receiver position separately
    rx_pos, h_meas, traced_paths = next_item[0], next_item[1], next_item[2:]

    # Build traced paths
    traced_paths = tensor_dicts_to_traced_paths(scene, traced_paths)

    # Tile
    traced_paths = pad_traced_paths(traced_paths, max_num_paths_spec, max_num_paths_diff, max_num_paths_scat)

    # Serialize tiled traced paths
    record_bytes = serialize_traced_paths(rx_pos, h_meas, traced_paths, False)

    # Save the tiled traced paths
    file_writer.write(record_bytes)

    # Print progress
    progress_message = f"\rProgress: {it_num+1}/{traced_paths_dataset_size}"
    progress_message = progress_message.ljust(line_length)
    print(progress_message, end="")

file_writer.close()
print("")


#########################################
# Removing the raw dataset
#########################################

# Removes the raw (unpost-processed) dataset if requested
if delete_raw_dataset:
    os.remove(traced_paths_raw_dataset_datafile)


#######################################
# Save the dataset properties
#######################################

# Filename for storing the dataset parameters
params_filename = os.path.join(traced_paths_dataset_folder, traced_paths_dataset + '.json')

# Retrieve the input parameters as a dict
params = vars(args)

# Add the maximum number of paths
params['max_num_paths_spec'] = max_num_paths_spec
params['max_num_paths_diff'] = max_num_paths_diff
params['max_num_paths_scat'] = max_num_paths_scat

# Dump the dataset parameters in a JSON file
with open(params_filename, "w") as outfile:
    json.dump(params, outfile)
