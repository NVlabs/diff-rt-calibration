# Learning Radio Environments by Differentiable Ray Tracing

This repository contains the source code to reproduce the results from the paper [Learning Radio Environments by Differentiable Ray Tracing [A]](https://arxiv.org/abs/2311.18558)
using the [Sionna&trade; link-level simulator [B]](https://nvlabs.github.io/sionna/) and its differentiable ray tracing extension [Sionna RT [C]](https://arxiv.org/abs/2303.11103).

## Abstract
Ray tracing (RT) is instrumental in 6G research in order to generate spatially-consistent and environment-specific channel impulse responses(CIRs). While acquiring accurate scene geometries is now relatively straightforward, determining material characteristics requires precise calibration using channel measurements. We therefore introduce a novel gradient-based calibration method, complemented by differentiable parametrizations of material properties, scattering and antenna patterns. Our method seamlessly integrates with differentiable ray tracers that enable the computation of derivatives of CIRs with respect to these parameters. Essentially, we approach field computation as a large computational graph wherein parameters are trainable akin to weights of a neural network (NN). We have validated our method using both synthetic data and real-world indoor channel measurements, employing a distributed multiple-input multiple-output (MIMO) channel sounder.

## Setup
Running this code requires [Sionna 0.16](https://nvlabs.github.io/sionna/) or later.
To run the notebooks on your machine, you also need [Jupyter](https://jupyter.org).
We recommend Ubuntu 22.04, Python 3.10, and TensorFlow 2.13.

## How to use this repository

The notebook [Synthetic_Data.ipynb](notebooks/Synthetic_Data.ipynb) reproduces all the results from Secion IV-B.
You don't need to download or generate any additional file to run this notebook.

To reproduce the results from Section IV-C, you first need to download the "dichasus-dc01.tfrecords" file from the [DICHASUS website](https://dichasus.inue.uni-stuttgart.de/datasets/data/dichasus-dcxx/) to the folder `data/tfrecords` within the cloned repository. More information about the DICHASUS channel sounder can be found in [[D]](https://arxiv.org/abs/2206.15302).

Then, you need to create a dataset of traced paths using the script [gen_dataset.py](code/gen_dataset.py).
For this purpose, ensure that you are in the `code/` folder, and run:

```bash
python gen_dataset.py -traced_paths_dataset dichasus-dc01 -traced_paths_dataset_size 10000
```
This script stores the generated dataset in the `data/traced_paths/` folder.
Generating the dataset of traced paths can take a while.

Once the dataset is generated, you can train the models considered in Section IV-C by running the corresponding notebooks ([ITU Materials](notebooks/ITU_Materials.ipynb), [Learned Materials](notebooks/Learned_Materials.ipynb) and [Neural Materials](notebooks/Neural_Materials.ipynb)).
The weights of the trained models are saved by the notebooks in the `checkpoints/` folder.
Note that for "ITU Materials", training consists in fitting the power scaling factor.

Once the trainings are done, the figures from Section IV-C can be reproduced by the notebooks [CDFs.ipynb](notebooks/CDFs.ipynb), [Heat_Maps.ipynb](notebooks/Heat_Maps.ipynb) and [CIRs](notebooks/CIRs.ipynb).

## Structure of this repository
    .
    ├── LICENSE.txt                            # License file
    ├── README.md                              # Readme
    ├── checkpoints                            # Folder to store weights after training
    ├── code
    │   ├── gen_dataset.py                     # Script to generate a dataset of traced paths
    │   ├── neural_materials.py                # NeuralMaterials Class
    │   ├── trainable_antenna_pattern.py       # TrainableAntennaPattern Class
    │   ├── trainable_materials.py             # TrainableMaterials Class
    │   ├── trainable_scattering_pattern.py    # TrainableScatteringPatterm Class
    │   └── utils.py                           # Utility functions
    ├── data
    │   ├── coordinates.csv                    # Coordinates of receivers and other points of interest
    │   ├── reftx-offsets-dichasus-dc01.json   # Phase and time offsets
    │   ├── spec.json                          # Specification of the "dichasus-dcxx" dataset
    │   ├── synthetic_positions.npy            # Positions used for the results with synthetic data
    │   ├── tfrecords                          # Folder to store DICHASUS tfrecords files
    │   └── traced_paths                       # Folder to store datasets of traced paths from `gen_dataset.py`
    ├── notebooks
    │   ├── CDFs.ipynb                         # CDFs from Section IV-C
    │   ├── CIRs.ipynb                         # CIRs from Section IV-C
    │   ├── Heat_Maps.ipynb                    # Heatmaps from Section IV-C
    │   ├── ITU_Materials.ipynb                # Fits the power scaling for the "ITU Materials" baseline from Section IV-C
    │   ├── Neural_Materials.ipynb             # Trains the "Neural Materials" model from Section IV-C
    │   ├── Synthetic_Data.ipynb               # Results from Section IV-B
    │   └── Learned_Materials.ipynb            # Trains the "Learned Materials" model from Section IV-C
    ├── results                                # Precomputed results and figures. Might be overwritten when notebooks are executed.
    │   ├── measurements                       # Results for measured data
    │   └── synthetic                          # Results for synthetic data
    └── scenes
        └── inue_simple                        # Simple 3D model of the INUE at Stuttgart University
            ├── inue_simple.blend              # Blender scene file
            ├── inue_simple.xml                # Mitsuba scene file
            └── meshes                         # Meshes used in the Mitsuba scene
                ├── ceiling.ply
                ├── floor.ply
                └── walls.ply

## References

[A] J. Hoydis, F. Ait Aoudia, S. Cammerer, F. Euchner, M. Nimier-David, S. ten Brink, A. Keller, ["Learning Radio Environments by Differentiable Ray Tracing"](https://arxiv.org/abs/2311.18558), Mar. 2023.

[B] J. Hoydis, S. Cammerer, F. Ait Aoudia, A. Vem, N. Binder, G. Marcus, A. Keller, ["Sionna: An Open-Source Library for Next-Generation Physical Layer Research"](https://arxiv.org/abs/2203.11854), Mar. 2022.

[C] J. Hoydis, F. Ait Aoudia, S. Cammerer, M. Nimier-David, N. Binder, G. Marcus, A. Keller, ["Sionna RT: Differentiable Ray Tracing for Radio Propagation Modeling"](https://arxiv.org/abs/2303.11103), Mar. 2023.

[D] F. Euchner,  M. Gauger, S. Doerner, S. ten Brink, ["A Distributed Massive MIMO Channel Sounder for "Big CSI Data"-driven Machine Learning"](https://arxiv.org/abs/2206.15302),
in Proc. Int. ITG Works. Smart Antennas (WSA), Nov. 2021.

## License and Citation

Copyright &copy; 2023, NVIDIA Corporation. All rights reserved.

This work is made available under the [NVIDIA License](LICENSE.txt).

If you use this software, please cite it as:
```bibtex
@article{sionna-rt-calibration,
    title = {{Learning Radio Environments by Differentiable Ray Tracing}},
    author = {Hoydis, Jakob and {Ait Aoudia}, Fayçal and Cammerer, Sebastian and Euchner, Florian, and Nimier-David, Merlin and ten Brink, Stephan, and Keller, Alexander},
    year = {2023},
    month = DEC,
    journal = {arXiv preprint},
    online = {https://arxiv.org/}
}
```
