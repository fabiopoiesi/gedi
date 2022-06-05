import torch
import torch.nn as nn
import torch.nn.functional as F
import torchgeometry as tgm
import numpy as np
import open3d.ml.torch as ml3d
from backbones.pointnet2_ops_lib.pointnet2_ops.pointnet2_modules import PointnetSAModule


class tnet(nn.Module):

    def __init__(self,):
        super(tnet, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv1d(3, 256, 1, bias=False),
                                   nn.BatchNorm1d(256),
                                   nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv1d(256, 512, 1, bias=False),
                                   nn.BatchNorm1d(512),
                                   nn.ReLU())

        self.conv3 = nn.Sequential(nn.Conv1d(512, 1024, 1, bias=False),
                                   nn.BatchNorm1d(1024))

        self.fc1 = nn.Sequential(nn.Linear(1024, 512, bias=False),
                                 nn.BatchNorm1d(512),
                                 nn.ReLU())

        self.fc2 = nn.Sequential(nn.Linear(512, 256, bias=False),
                                 nn.BatchNorm1d(256),
                                 nn.ReLU())
        self._init_last_layer()

    def _init_last_layer(self):
        self.fc3 = nn.Linear(256, 9, bias=True)
        torch.nn.init.zeros_(self.fc3.bias)

    def _forward_last_layer(self, x):
        x = self.fc3(x)
        x = x + torch.eye(3, device='cuda').view(1, 9).repeat(x.size()[0], 1)
        x = x.view(-1, 3, 3)
        return x

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, x.shape[1])

        x = self.fc1(x)
        x = self.fc2(x)
        x = self._forward_last_layer(x)

        return x


class qnet(tnet):

    def _init_last_layer(self):
        self.fc3 = nn.Linear(256, 4, bias=True)
        torch.nn.init.zeros_(self.fc3.bias)

    def _forward_last_layer(self, x):
        quat = self.fc3(x)
        quat = quat + torch.tensor([1, 0, 0, 0], device='cuda').repeat(quat.size()[0], 1)
        quat = F.normalize(quat, p=2, dim=1)
        return quat


class PointNet2Feature(nn.Module):

    def __init__(self, dim=32):
        super(PointNet2Feature, self).__init__()

        self.use_xyz = True
        self.qnet = qnet()

        self.samodule1 = PointnetSAModule(
            npoint=128,
            radius=0.2,
            nsample=32,
            mlp=[3, 128, 128, 128],
            use_xyz=self.use_xyz,
        )

        self.samodule2 = PointnetSAModule(
            npoint=64,
            radius=0.4,
            nsample=16,
            mlp=[128+3, 256, 256, 256],
            use_xyz=self.use_xyz,
        )

        self.samodule3 = PointnetSAModule(
            mlp=[256+3, 512, 512, 1024], use_xyz=self.use_xyz
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(256, dim),
        )

    def _forward(self, pc):

        quat = self.qnet(pc)
        angle_axis = tgm.quaternion_to_angle_axis(quat)
        _trans = tgm.angle_axis_to_rotation_matrix(angle_axis)
        trans = _trans[:, :3, :3]
        pc = trans @ pc

        xyz = pc.transpose(1, 2).contiguous()
        xyz, features = self.samodule1(xyz, None)
        xyz, features = self.samodule2(xyz, features)
        xyz, features = self.samodule3(xyz, features)
        out = self.fc_layer(features.squeeze(-1))
        out = F.normalize(out, p=2, dim=1)

        return out, pc, trans

    def forward(self, xa, xp=torch.Tensor([])):
        if xp.nelement() == 0:
            f, _, _ = self._forward(xa)
            return f
        else:
            f0, pc0, trans0 = self._forward(xa)
            f1, pc1, trans1 = self._forward(xp)
            return f0, pc0, trans0, f1, pc1, trans1


class LRF(nn.Module):

    def __init__(self, patches_per_pair=256, samples_per_patch=256, eps=1e-12, r_lrf=1, device='cpu'):
        super(LRF, self).__init__()

        self.eps = eps
        self.r_lrf = r_lrf
        self.patches_per_pair = patches_per_pair
        self.samples_per_patch = samples_per_patch
        self.device = device

    def _forward(self, xp, xpi):

        B, N, c = xpi.size()
        xpi = xpi.contiguous()  # dim = B x 3 x N
        xp = xp.unsqueeze(2).contiguous()  # dim = B x 3 x 1

        # zp
        x = xp - xpi  # pi->p = p - pi
        xxt = torch.bmm(x, x.transpose(1, 2)) / c

        _, _, v = torch.svd(xxt.to(self.device))
        v = v.to(self.device)

        with torch.no_grad():
            sum_ = (v[..., -1].unsqueeze(1) @ x).sum(2)
            _sign = torch.ones((len(xpi), 1), device=self.device) - 2 * (sum_ < 0)

        zp = (_sign * v[..., -1]).unsqueeze(1)  # B x 1 x 3

        # xp
        x *= -1  # p->pi = pi - p
        norm = (zp @ x).transpose(1, 2)
        proj = norm * zp

        vi = x - proj.transpose(1, 2)

        x_l2 = torch.sqrt((x ** 2).sum(dim=1, keepdim=True))

        alpha = self.r_lrf - x_l2
        alpha = alpha * alpha
        beta = (norm * norm).transpose(1, 2)
        vi_c = (alpha * beta * vi).sum(2)

        xp = (vi_c / torch.sqrt((vi_c ** 2).sum(1, keepdim=True)))

        # yp
        yp = torch.cross(xp, zp.squeeze(), dim=1)

        lrf = torch.cat((xp.unsqueeze(2), yp.unsqueeze(2), zp.transpose(1, 2)), dim=2)

        return lrf

    def forward(self, x0, x0i, x1=None, x1i=None):

        # compute local reference frames
        lrf0 = self._forward(x0, x0i)
        inds = np.random.choice(x0i.shape[2], self.samples_per_patch, replace=False)
        _out_x0 = (x0i[..., inds] - x0.unsqueeze(-1)) / self.r_lrf
        out_x0 = lrf0.transpose(1, 2) @ _out_x0

        if x1 is None:
            return out_x0

        lrf1 = self._forward(x1, x1i)
        inds = np.random.choice(x1i.shape[2], self.samples_per_patch, replace=False)
        _out_x1 = (x1i[..., inds] - x1.unsqueeze(-1)) / self.r_lrf
        out_x1 = lrf1.transpose(1, 2) @ _out_x1

        return out_x0, out_x1


class GeDi:

    def __init__(self, config):
        self.dim = config['dim']
        self.samples_per_batch = config['samples_per_batch']
        self.samples_per_patch_lrf = config['samples_per_patch_lrf']
        self.samples_per_patch_out = config['samples_per_patch_out']
        self.r_lrf = config['r_lrf']

        self.lrf = LRF(patches_per_pair=self.samples_per_batch,
                       samples_per_patch=self.samples_per_patch_out,
                       r_lrf=self.r_lrf,
                       device='cpu')

        self.gedi_net = PointNet2Feature(dim=self.dim)
        self.gedi_net.load_state_dict(torch.load(config['fchkpt_gedi_net'])['pnet_model_state_dict'])
        self.gedi_net.cuda().eval()

    def compute(self, pts, pcd):

        radii = self.r_lrf * torch.ones((len(pts)))

        out = ml3d.ops.radius_search(pcd, pts, radii,
                                     points_row_splits=torch.LongTensor([0, len(pcd)]),
                                     queries_row_splits=torch.LongTensor([0, len(pts)]))

        pcd_desc = np.empty((len(pts), self.dim))

        for b in range(int(np.ceil(len(pts) / self.samples_per_batch))):

            i_start = b * self.samples_per_batch
            i_end = (b + 1) * self.samples_per_batch
            if i_end > len(pts):
                i_end = len(pts)

            x = np.empty((i_end - i_start, 3, self.samples_per_patch_lrf))

            j = 0
            for i in range(i_start, i_end):

                _inds = out[0][out[1][i]:out[1][i + 1]]
                try:
                    inds = np.random.choice(_inds.numpy(), size=self.samples_per_patch_lrf, replace=False)
                except:
                    # print('[w] got patch with few points -> {}. Padding with replicas ...'.format(len(pt_nn)))
                    inds = np.r_[_inds, np.random.choice(_inds.numpy(), self.samples_per_patch_lrf - len(_inds))]

                x[j] = pcd[inds].T

                j += 1

            x = torch.Tensor(x)

            patch = self.lrf(pts[i_start:i_end], x)

            with torch.no_grad():
                f = self.gedi_net(patch.cuda())

            pcd_desc[i_start:i_end] = f.cpu().detach().numpy()[:i_end - i_start]

        return pcd_desc


if __name__ == '__main__':

        x = torch.rand((100, 3, 512)).cuda()  # [npatches, xyz, npoints]
        net = PointNet2Feature().cuda().eval()
        out = net(x)
        print(out.shape)
