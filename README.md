# GeDi: Learning general and distinctive 3D local deep descriptors for point cloud registration - IEEE T-PAMI

Official repository of GeDi descriptor. [Paper (pdf)](https://arxiv.org/pdf/2105.10382.pdf)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/generalisable-and-distinctive-3d-local-deep/point-cloud-registration-on-3dmatch-benchmark)](https://paperswithcode.com/sota/point-cloud-registration-on-3dmatch-benchmark?p=generalisable-and-distinctive-3d-local-deep)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/generalisable-and-distinctive-3d-local-deep/point-cloud-registration-on-3dmatch-trained)](https://paperswithcode.com/sota/point-cloud-registration-on-3dmatch-trained?p=generalisable-and-distinctive-3d-local-deep)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/generalisable-and-distinctive-3d-local-deep/point-cloud-registration-on-eth-trained-on)](https://paperswithcode.com/sota/point-cloud-registration-on-eth-trained-on?p=generalisable-and-distinctive-3d-local-deep)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/generalisable-and-distinctive-3d-local-deep/point-cloud-registration-on-kitti)](https://paperswithcode.com/sota/point-cloud-registration-on-kitti?p=generalisable-and-distinctive-3d-local-deep)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/generalisable-and-distinctive-3d-local-deep/point-cloud-registration-on-kitti-trained-on)](https://paperswithcode.com/sota/point-cloud-registration-on-kitti-trained-on?p=generalisable-and-distinctive-3d-local-deep)

<p align="center"><img src="resources/training_scheme.png" width="1000"></p>

| 3DMatch ⟶ ETH        | 3DMatch ⟶ KITTI           | KITTI ⟶ 3DMatch
|:---------------------------:|:---------------------------:|:---------------------------:|
| ![](resources/3dm_eth.png) | ![](resources/3dm_kitti.png) | ![](resources/kitti_3dm.png) |

An effective 3D descriptor should be invariant to different geometric transformations, such as scale and rotation, robust to occlusions and clutter, and capable of generalising to different application domains.
**We present a simple yet effective method to learn general and distinctive 3D local descriptors (GeDi) that can be used to register point clouds that are captured in different domains.**
Point cloud patches are extracted, canonicalised with respect to their local reference frame, and encoded into scale and rotation-invariant compact descriptors by a deep neural network that is invariant to permutations of the input points.
This design is what enables GeDi to generalise across domains.

Gedi is an extension of [DIP](https://arxiv.org/abs/2009.00258) descriptor. Additional code and data can be found in [DIP repository](https://github.com/fabiopoiesi/dip).

## Tested with

- Ubuntu 20.04
- CUDA 11.x
- Python 3.8
- PyTorch 1.8.1
- [Open3D 0.15.1](http://www.open3d.org/docs/release/)
- [torchgeometry v0.1.2](https://kornia.readthedocs.io/en/v0.1.2/)

## Installation

Set up your environment and start it 

```
virtualenv venv -p python3
source venv/bin/activate
pip install --upgrade pip
```

In order to have Open3D functioning correctly on GPU, you should download and install this version of `torch` ([link](https://github.com/isl-org/open3d_downloads/releases/tag/torch1.8.1)). Once you have downloaded it, install it as

```
pip install torch-1.8.1-cp38-cp38-linux_x86_64.whl
```

Then install the following packages
```
pip install open3d==0.15.1
pip install torchgeometry==0.1.2
pip install gdown
pip install tensorboard
pip install protobuf==3.20
```
Note: the last pip is to downgrade protobuf in order to avoid your system complaining that the version is too high.

Now, you have to compile and install [Pointnet2](https://github.com/erikwijmans/Pointnet2_PyTorch). This may be a bit tricky, as it could complain if the version of CUDA is different from that of PyTorch. You can check the [original repository](https://github.com/erikwijmans/Pointnet2_PyTorch) for more information.

```
cd backbones
pip install ./pointnet2_ops_lib/
```

## Download data

The script `download_data.py` will download the **pretained model** and **assets** to run the demo. Data will be put in the right directories automatically. The model was trained on [3DMatch](http://3dmatch.cs.princeton.edu/) training set.

```
python download_data.py
```

## Demo

Once the data are downloaded, execute the demo as

```
python demo.py
```

The result will look like these (note that results may slightly differ from run to run due to the randomisation of RANSAC).


| Before registration           | After registration           |
|:---------------------------:|:---------------------------:|
| <img src="resources/demo0.png" width="500"> | <img src="resources/demo1.png" width="500"> |


## Citing our work

Please cite the following paper if you use our code

```latex
@inproceedings{Poiesi2021,
  title = {Learning general and distinctive 3D local deep descriptors for point cloud registration},
  author = {Poiesi, Fabio and Boscaini, Davide},
  booktitle = {IEEE Trans. on Pattern Analysis and Machine Intelligence},
  year = {(early access) 2022}
}
```

## Acknowledgements

This research was supported by the [SHIELD project](http://shield.cyi.ac.cy/), funded by the European Union’s Joint Programming Initiative – Cultural Heritage, Conservation, Protection and Use joint call ([link](https://www.heritageresearch-hub.eu/homepage/joint-programming-initiative-on-cultural-heritage-homepage/)), and partially by Provincia Autonoma di Trento (Italy) under L.P. 6/99 as part of the X-Loader4.0 project ([link](https://tev.fbk.eu/projects/xloader4)).

## TODOS
- Add test code
- Add training code
- Add support for [ETH](https://projects.asl.ethz.ch/datasets/doku.php?id=laserregistration:laserregistration) and [KITTI](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) datasets
