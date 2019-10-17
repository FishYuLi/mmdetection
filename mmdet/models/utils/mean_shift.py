import torch
import torch.nn as nn
import torchvision as tv
import numpy as np

STOP_THRESHOLD = 1e-4
CLUSTER_THRESHOLD = 1e-1

import pdb

class MeanShift(object):
    def __init__(self, bandwidth, stop_thr=1e-4, clus_thr=1e-3):

        self.bandwidth = bandwidth
        self.cons_plus = torch.from_numpy(np.array([1 / (bandwidth * np.sqrt(2 * np.pi))])).cuda().float()
        self.stop_thr = stop_thr
        self.clus_thr = clus_thr

    def kernel(self, dist):

        inner = dist / self.bandwidth
        inner = inner * inner
        res = self.cons_plus * torch.exp(-0.5 * inner)

        return res

    def distance(self, A, B, keepdim=False):
        # L2 Norm
        dis = torch.norm(A - B, dim=-1, keepdim=keepdim)
        # ================

        # IOU
        #
        return dis

    def mean_shift(self, points, stay):

        p_left = points.clone()

        n = points.shape[0]
        c = p_left.shape[0]

        shift_points = []

        corresponding_points = []
        points_copy = stay.clone()
        while c > 0:

            # # Using L2 distance
            p_mesh = p_left.unsqueeze(1) # (c, 5) => (c, 1, 5)
            p_mesh = p_mesh.expand(-1, n, -1) # (c, 1, 5) => (c, n, 5)
            orip_mesh = points.unsqueeze(0) # (n, 5) => (1, n, 5)
            orip_mesh = orip_mesh.expand(c, -1, -1) # (1, n, 5) => (c, n, 5)
            # dist = self.distance(p_mesh, orip_mesh, keepdim=True) # (c, n, 1)

            # Using IoU
            dist = tv.ops.box_iou(p_left, points)
            dist = 1 - dist.unsqueeze(2)

            weight = self.kernel(dist) # (c, n, 1)

            p_shift = orip_mesh * weight # (c, n, 5)
            p_shift = p_shift.sum(dim=1) / weight.sum(dim=1) # (c, 5)

            shift = self.distance(p_left, p_shift)  # (c, )

            left_idx = (shift > self.stop_thr).nonzero(as_tuple=True)[0] # (newc, )
            save_idx = (shift <= self.stop_thr).nonzero(as_tuple=True)[0] # (c - newc, )

            p_left = p_shift.index_select(dim=0, index=left_idx)
            c = p_left.shape[0]

            corresponding_points.append(points_copy.index_select(dim=0, index=save_idx))
            points_copy = points_copy.index_select(dim=0, index=left_idx)
            shift_points.append(p_shift.index_select(dim=0, index=save_idx))

        shift_points = torch.cat(shift_points, 0)
        corresponding_points = torch.cat(corresponding_points, 0)
        cluster_ids, cluster_centers = self._cluster_points(shift_points)

        return shift_points, cluster_ids, cluster_centers, corresponding_points

    def _cluster_points(self, shift_points):

        cluster_idx = 0
        cluster_centers = []

        n = shift_points.shape[0]
        p = shift_points.unsqueeze(1) # (n, 1, 5)
        p = p.expand(-1, n, -1)
        pt = shift_points.unsqueeze(0) # (1, n, 5)
        pt = pt.expand(n, -1, -1)

        dist = self.distance(p, pt)  # (n, n)

        is_center = torch.ones((n, ))
        cluster_ids = torch.ones((n, )) * -1

        for i in range(n):

            if is_center[i] == 0:
                continue

            cluster_centers.append(shift_points[i])
            cur_dist = dist[i]

            same_cluster_idx = (cur_dist < self.clus_thr).nonzero(as_tuple=True)[0]

            cluster_ids[same_cluster_idx] = cluster_idx
            cluster_idx += 1

            is_center[same_cluster_idx] = 0

            dist[:, same_cluster_idx] = 10000

        cluster_centers = torch.stack(cluster_centers, dim=0)

        return cluster_ids, cluster_centers
