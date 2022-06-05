import torch
import numpy as np
import open3d as o3d
from gedi import GeDi

'''
demo to show the registration between two point clouds using GeDi descriptors
- the first visualisation shows the two point clouds in their original reference frame
- the second visualisation show point cloud 0 transformed in the reference frame of point cloud 1
'''


config = {'dim': 32,                                            # descriptor output dimension
          'samples_per_batch': 500,                             # batches to process the data on GPU
          'samples_per_patch_lrf': 4000,                        # num. of point to process with LRF
          'samples_per_patch_out': 512,                         # num. of points to sample for pointnet++
          'r_lrf': .5,                                          # LRF radius
          'fchkpt_gedi_net': 'data/chkpts/3dmatch/chkpt.tar'}   # path to checkpoint

voxel_size = .01
patches_per_pair = 5000

# initialising class
gedi = GeDi(config=config)

# getting a pair of point clouds
pcd0 = o3d.io.read_point_cloud('data/assets/threed_match_7-scenes-redkitchen_cloud_bin_0.ply')
pcd1 = o3d.io.read_point_cloud('data/assets/threed_match_7-scenes-redkitchen_cloud_bin_5.ply')

pcd0.paint_uniform_color([1, 0.706, 0])
pcd1.paint_uniform_color([0, 0.651, 0.929])

# estimating normals (only for visualisation)
pcd0.estimate_normals()
pcd1.estimate_normals()

o3d.visualization.draw_geometries([pcd0, pcd1])

# randomly sampling some points from the point cloud
inds0 = np.random.choice(np.asarray(pcd0.points).shape[0], patches_per_pair, replace=False)
inds1 = np.random.choice(np.asarray(pcd1.points).shape[0], patches_per_pair, replace=False)

pts0 = torch.tensor(np.asarray(pcd0.points)[inds0]).float()
pts1 = torch.tensor(np.asarray(pcd1.points)[inds1]).float()

# applying voxelisation to the point cloud
pcd0 = pcd0.voxel_down_sample(voxel_size)
pcd1 = pcd1.voxel_down_sample(voxel_size)

_pcd0 = torch.tensor(np.asarray(pcd0.points)).float()
_pcd1 = torch.tensor(np.asarray(pcd1.points)).float()

# computing descriptors
pcd0_desc = gedi.compute(pts=pts0, pcd=_pcd0)
pcd1_desc = gedi.compute(pts=pts1, pcd=_pcd1)

# preparing format for open3d ransac
pcd0_dsdv = o3d.pipelines.registration.Feature()
pcd1_dsdv = o3d.pipelines.registration.Feature()

pcd0_dsdv.data = pcd0_desc.T
pcd1_dsdv.data = pcd1_desc.T

_pcd0 = o3d.geometry.PointCloud()
_pcd0.points = o3d.utility.Vector3dVector(pts0)
_pcd1 = o3d.geometry.PointCloud()
_pcd1.points = o3d.utility.Vector3dVector(pts1)

# applying ransac
est_result01 = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    _pcd0,
    _pcd1,
    pcd0_dsdv,
    pcd1_dsdv,
    mutual_filter=True,
    max_correspondence_distance=.02,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
    ransac_n=3,
    checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(.9),
              o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(.02)],
    criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1000))

# applying estimated transformation
pcd0.transform(est_result01.transformation)
o3d.visualization.draw_geometries([pcd0, pcd1])
