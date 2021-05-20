# Generalisable and distinctive 3D local deep descriptors (GeDi) for point cloud registration

| 3DMatch ⟶ ETH        | 3DMatch ⟶ KITTI           | KITTI ⟶ 3DMatch
|:---------------------------:|:---------------------------:|:---------------------------:|
| ![](assets/3dm_eth.png) | ![](assets/3dm_kitti.png) | ![](assets/kitti_3dm.png) |

## Description

An effective 3D descriptor should be invariant to different geometric transformations, such as scale and rotation, repeatable in the case of occlusions and clutter, and generalisable in different contexts when data is captured with different sensors.
GeDi is a simple but yet effective method to learn generalisable and distinctive 3D local descriptors that can be used to register point clouds captured in different contexts with different sensors.
Point cloud patches are extracted, canonicalised with respect to their local reference frame, and encoded into scale and rotation-invariant compact descriptors by a PointNet++-based deep neural network.
Our descriptors can effectively generalise across different sensor modalities from locally and randomly sampled points.
The graphs above show the comparison between our descriptors and state-of-the-art descriptors on several indoor and outdoor datasets reconstructed using both RGBD sensors and laser scanners.
In particular, [3DMatch](https://3dmatch.cs.princeton.edu/) is an indoor dataset captured with RGBD sensors, [ETH](https://projects.asl.ethz.ch/datasets/doku.php?id=laserregistration:laserregistration) and [KITTI](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) are outdoor datasets captured with laser scanners.
Our descriptors outperform most recent descriptors by a large margin in terms of generalisation, and become the state of the art also in benchmarks where training and testing are performed in the same scenarios.