# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
from ..da_heads.da_heads import build_da_heads
from ..roi_heads.box_head.roi_box_feature_extractors import make_roi_box_feature_extractor


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()
        self.cfg = cfg.clone()
        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg)
        self.roi_heads = build_roi_heads(cfg)
        self.da_heads = build_da_heads(cfg)
        # self.mc_heads = build_mc_heads(cfg)
        self.feature_extractor = make_roi_box_feature_extractor(cfg)


    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList ] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        # proposals, proposal_losses = self.rpn(images, features, targets)
        if self.training:
            proposals, proposal_losses, objectness = self.rpn(images, features, targets)
        else:
            proposals, proposal_losses = self.rpn(images, features, targets)

        da_losses = {}
        if self.roi_heads:
            # x, result, detector_losses, da_ins_feas, da_ins_labels = self.roi_heads(features, proposals, targets)
            if self.training:
                x, result, detector_losses, da_ins_feas, da_ins_labels, class_logits, da_proposals, mask_feas = self.roi_heads(features, proposals, targets)
                cls_prob = F.softmax(class_logits, dim=1)
                if self.da_heads:
                    ### Image-level Uncertainty-aware Adversarial Learning
                    if self.cfg.MODEL.ENT_WEIGHTED_ON_RPN:
                        rpn_cls = torch.sigmoid(objectness[0])
                        rpn_cls_ent = -(torch.mul(rpn_cls, torch.log2(rpn_cls + 1e-30)) +
                                        torch.mul((1 - rpn_cls),torch.log2((1 - rpn_cls) + 1e-30))) / np.log2(2)
                        da_img_weight_ori, _ = torch.min(rpn_cls_ent, 1)
                        da_img_weight_max = da_img_weight_ori.max().detach()
                        da_img_weight = (da_img_weight_ori + da_img_weight_max) / 2
                    else:
                        da_img_weight = None

                    ### Instance-level Uncertainty-aware Adversarial Learning
                    if self.cfg.MODEL.ENT_WEIGHTED_ON:
                        cls_prob_ent = -torch.sum(
                            torch.mul(cls_prob, torch.log2(cls_prob + 1e-30)) / np.log2(cls_prob.shape[1]), 1)
                        da_ins_weight = cls_prob_ent.clone()
                        da_ins_weight = da_ins_weight * 10
                        da_ins_weight[da_ins_weight > 1] = 1
                        da_ins_weight[:256] = (da_ins_weight[256:]).mean()

                        ### nstance-level Uncertainty-guided Curriculum Learning
                        if self.cfg.MODEL.ENT_WEIGHTED_ON_RCNN_INTER:
                            if not self.cfg.MODEL.ENT_WEIGHTED_ON_RPN:
                                rpn_cls = torch.sigmoid(objectness[0])
                                rpn_cls_ent = -(torch.mul(rpn_cls, torch.log2(rpn_cls + 1e-30)) +
                                                torch.mul((1 - rpn_cls), torch.log2((1 - rpn_cls) + 1e-30))) / np.log2(2)
                                da_img_weight_ori, _ = torch.min(rpn_cls_ent, 1)
                            from maskrcnn_benchmark.modeling.poolers import Pooler
                            scales = self.cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
                            sampling_ratio = self.cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
                            pooler = Pooler(output_size=(1, 1), scales=scales, sampling_ratio=sampling_ratio, )
                            rpn_cls_ent_roi_pool = pooler([da_img_weight_ori.unsqueeze(1)], da_proposals)
                            rpn_inter_ent = rpn_cls_ent_roi_pool.squeeze(1).squeeze(1).squeeze(1)
                            rpn_inter_weights = rpn_inter_ent.detach() < 0.5
                            da_ins_weight = rpn_inter_weights.float() * da_ins_weight
                    else:
                        da_ins_weight = None

                    da_losses = self.da_heads(features, da_ins_feas, da_ins_labels, targets, da_img_weight,
                                              da_ins_weight, cls_prob, mask_feas)
            else:
                x, result, detector_losses, da_ins_feas, da_ins_labels, mask_feas = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            losses.update(da_losses)
            return losses

        return result
