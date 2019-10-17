from mmdet.core import (bbox2roi, bbox_mapping, merge_aug_bboxes,
                        merge_aug_masks, merge_aug_proposals, multiclass_nms)
import torch
import pdb
import numpy as np

import os
import cv2

DRAW=False

class RPNTestMixin(object):

    def simple_test_rpn(self, x, img_meta, rpn_test_cfg):
        rpn_outs = self.rpn_head(x)
        proposal_inputs = rpn_outs + (img_meta, rpn_test_cfg)
        proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        return proposal_list

    def aug_test_rpn(self, feats, img_metas, rpn_test_cfg):
        imgs_per_gpu = len(img_metas[0])
        aug_proposals = [[] for _ in range(imgs_per_gpu)]
        for x, img_meta in zip(feats, img_metas):
            proposal_list = self.simple_test_rpn(x, img_meta, rpn_test_cfg)
            for i, proposals in enumerate(proposal_list):
                aug_proposals[i].append(proposals)
        # reorganize the order of 'img_metas' to match the dimensions
        # of 'aug_proposals'
        aug_img_metas = []
        for i in range(imgs_per_gpu):
            aug_img_meta = []
            for j in range(len(img_metas)):
                aug_img_meta.append(img_metas[j][i])
            aug_img_metas.append(aug_img_meta)
        # after merging, proposals will be rescaled to the original image size
        merged_proposals = [
            merge_aug_proposals(proposals, aug_img_meta, rpn_test_cfg)
            for proposals, aug_img_meta in zip(aug_proposals, aug_img_metas)
        ]
        return merged_proposals


class BBoxTestMixin(object):

    def draw_cluster_bbox(self, img_file_name, draw_list):

        im = cv2.imread(img_file_name)
        alpha = 0.7
        im_copy = im.copy()
        all_colors = []
        centers = []
        for item in draw_list:
            ori_bbox, clus_id, cluster_centers, shift_bbox = item
            ori_bbox = ori_bbox.cpu().numpy()
            # ori_bbox = shift_bbox.cpu().numpy()
            cluster_centers = cluster_centers.cpu().numpy()
            centers.append(cluster_centers)
            clus_id = clus_id.cpu().numpy()

            bbox_num = ori_bbox.shape[0]

            cluster_num = cluster_centers.shape[0]
            colors = [np.random.randint(255, size=(3,)) for i in range(cluster_num)]
            colors = [c.tolist() for c in colors]
            all_colors.append(colors)

            for i in range(bbox_num):
                ori = ori_bbox[i]
                cluster = int(clus_id[i])

                pt1 = (int(ori[0]), int(ori[1]))
                pt2 = (int(ori[2]), int(ori[3]))
                # score = int(ori[4])

                color = colors[cluster]
                cv2.rectangle(im, pt1, pt2, color, thickness=1)

        im = im * alpha + im_copy * (1-alpha)
        for colors, cluster_centers in zip(all_colors, centers):
            cluster_num = cluster_centers.shape[0]
            for i in range(cluster_num):
                ori = cluster_centers[i]

                pt1 = (int(ori[0]), int(ori[1]))
                pt2 = (int(ori[2]), int(ori[3]))
                # score = int(ori[4])

                color = colors[i]
                cv2.rectangle(im, pt1, pt2, (255,255,255), thickness=3)
                cv2.rectangle(im, pt1, pt2, color, thickness=2)

        name = img_file_name.split('/')[-1]
        save_path = os.path.join('./draw', name)
        cv2.imwrite(save_path, im)

    def get_dets_from_shift(self, shift_points, cluster_ids, cluster_centers, ori_points, box_scale_fac):

        cluster_centers[:, :4] = cluster_centers[:, :4] * box_scale_fac
        cls_dets = torch.cat([cluster_centers, torch.ones((cluster_centers.shape[0],1)).cuda()], dim=1)

        shift_points[:, :4] = shift_points[:, :4] * box_scale_fac
        ori_points[:, :4] = ori_points[:, :4] * box_scale_fac
        return cls_dets, shift_points, ori_points

    def mean_shift_bbox(self, multi_bboxes, multi_scores, score_thr, max_num, roi_feats, img_shape, img_file_name):

        # box_scale_fac = np.array([img_shape[0], img_shape[1], img_shape[0], img_shape[1]])
        box_scale_fac = np.array([1., 1., 1., 1.])
        box_scale_fac = torch.from_numpy(box_scale_fac).float().cuda()
        num_classes = multi_scores.shape[1]
        bboxes, labels = [], []

        draw_list = []
        for i in range(1, num_classes):
            cls_inds = multi_scores[:, i] > score_thr
            if not cls_inds.any():
                continue
            # get bboxes and scores of this class
            if multi_bboxes.shape[1] == 4:
                _bboxes = multi_bboxes[cls_inds, :]
            else:
                _bboxes = multi_bboxes[cls_inds, i * 4:(i + 1) * 4]
            _bboxes = _bboxes / box_scale_fac
            _scores = multi_scores[cls_inds, i]
            cls_dets = torch.cat([_bboxes, _scores[:, None]], dim=1)

            shift_points, cluster_ids, cluster_centers, ori_points = self.meanshift.mean_shift(_bboxes, cls_dets)

            cls_dets, shift_points, ori_points = self.get_dets_from_shift(shift_points, cluster_ids, cluster_centers, ori_points, box_scale_fac)

            draw_list.append((ori_points, cluster_ids, cluster_centers, shift_points))
            cls_labels = multi_bboxes.new_full((cls_dets.shape[0], ),
                                               i - 1,
                                               dtype=torch.long)

            bboxes.append(cls_dets)
            labels.append(cls_labels)

        if DRAW:
            self.draw_cluster_bbox(img_file_name, draw_list)

        if bboxes:
            bboxes = torch.cat(bboxes)
            labels = torch.cat(labels)
            if bboxes.shape[0] > max_num:
                _, inds = bboxes[:, -1].sort(descending=True)
                inds = inds[:max_num]
                bboxes = bboxes[inds]
                labels = labels[inds]
        else:
            bboxes = multi_bboxes.new_zeros((0, 5))
            labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)

        return bboxes, labels

    def simple_test_bboxes(self,
                           x,
                           img_meta,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation."""
        rois = bbox2roi(proposals)
        roi_feats = self.bbox_roi_extractor(
            x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        if self.with_shared_head:
            roi_feats = self.shared_head(roi_feats)
        cls_score, bbox_pred = self.bbox_head(roi_feats)
        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        img_file_name = img_meta[0]['filename']
        # det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
        bboxes, scores = self.bbox_head.get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=None)
        # rcnn_test_cfg)
        det_bboxes, det_labels = self.mean_shift_bbox(bboxes, scores,
                                                      rcnn_test_cfg.score_thr,
                                                      rcnn_test_cfg.max_per_img,
                                                      roi_feats, img_shape,
                                                      img_file_name)

        return det_bboxes, det_labels

    def aug_test_bboxes(self, feats, img_metas, proposal_list, rcnn_test_cfg):
        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            # TODO more flexible
            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip)
            rois = bbox2roi([proposals])
            # recompute feature maps to save GPU memory
            roi_feats = self.bbox_roi_extractor(
                x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
            if self.with_shared_head:
                roi_feats = self.shared_head(roi_feats)
            cls_score, bbox_pred = self.bbox_head(roi_feats)
            bboxes, scores = self.bbox_head.get_det_bboxes(
                rois,
                cls_score,
                bbox_pred,
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)
        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores,
                                                rcnn_test_cfg.score_thr,
                                                rcnn_test_cfg.nms,
                                                rcnn_test_cfg.max_per_img)
        return det_bboxes, det_labels


class MaskTestMixin(object):

    def simple_test_mask(self,
                         x,
                         img_meta,
                         det_bboxes,
                         det_labels,
                         rescale=False):
        # image shape of the first image in the batch (only one)
        ori_shape = img_meta[0]['ori_shape']
        scale_factor = img_meta[0]['scale_factor']
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes - 1)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            _bboxes = (
                det_bboxes[:, :4] * scale_factor if rescale else det_bboxes)
            mask_rois = bbox2roi([_bboxes])
            mask_feats = self.mask_roi_extractor(
                x[:len(self.mask_roi_extractor.featmap_strides)], mask_rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
            mask_pred = self.mask_head(mask_feats)
            segm_result = self.mask_head.get_seg_masks(mask_pred, _bboxes,
                                                       det_labels,
                                                       self.test_cfg.rcnn,
                                                       ori_shape, scale_factor,
                                                       rescale)
        return segm_result

    def aug_test_mask(self, feats, img_metas, det_bboxes, det_labels):
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes - 1)]
        else:
            aug_masks = []
            for x, img_meta in zip(feats, img_metas):
                img_shape = img_meta[0]['img_shape']
                scale_factor = img_meta[0]['scale_factor']
                flip = img_meta[0]['flip']
                _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape,
                                       scale_factor, flip)
                mask_rois = bbox2roi([_bboxes])
                mask_feats = self.mask_roi_extractor(
                    x[:len(self.mask_roi_extractor.featmap_strides)],
                    mask_rois)
                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
                mask_pred = self.mask_head(mask_feats)
                # convert to numpy array to save memory
                aug_masks.append(mask_pred.sigmoid().cpu().numpy())
            merged_masks = merge_aug_masks(aug_masks, img_metas,
                                           self.test_cfg.rcnn)

            ori_shape = img_metas[0][0]['ori_shape']
            segm_result = self.mask_head.get_seg_masks(
                merged_masks,
                det_bboxes,
                det_labels,
                self.test_cfg.rcnn,
                ori_shape,
                scale_factor=1.0,
                rescale=False)
        return segm_result
