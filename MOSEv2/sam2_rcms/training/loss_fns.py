# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Dict, List

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F

from training.trainer import CORE_LOSS_KEY

from training.utils.distributed import get_world_size, is_dist_avail_and_initialized


def dice_loss(inputs, targets, num_objects, loss_on_multimask=False):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_objects: Number of objects in the batch
        loss_on_multimask: True if multimask prediction is enabled
    Returns:
        Dice loss tensor
    """
    inputs = inputs.sigmoid()
    if loss_on_multimask:
        # inputs and targets are [N, M, H, W] where M corresponds to multiple predicted masks
        assert inputs.dim() == 4 and targets.dim() == 4
        # flatten spatial dimension while keeping multimask channel dimension
        inputs = inputs.flatten(2)
        targets = targets.flatten(2)
        numerator = 2 * (inputs * targets).sum(-1)
    else:
        inputs = inputs.flatten(1)
        numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects


def sigmoid_focal_loss(
    inputs,
    targets,
    num_objects,
    alpha: float = 0.25,
    gamma: float = 2,
    loss_on_multimask=False,
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_objects: Number of objects in the batch
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        loss_on_multimask: True if multimask prediction is enabled
    Returns:
        focal loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if loss_on_multimask:
        # loss is [N, M, H, W] where M corresponds to multiple predicted masks
        assert loss.dim() == 4
        return loss.flatten(2).mean(-1) / num_objects  # average over spatial dims
    return loss.mean(1).sum() / num_objects


def iou_loss(
    inputs, targets, pred_ious, num_objects, loss_on_multimask=False, use_l1_loss=False
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        pred_ious: A float tensor containing the predicted IoUs scores per mask
        num_objects: Number of objects in the batch
        loss_on_multimask: True if multimask prediction is enabled
        use_l1_loss: Whether to use L1 loss is used instead of MSE loss
    Returns:
        IoU loss tensor
    """
    assert inputs.dim() == 4 and targets.dim() == 4
    pred_mask = inputs.flatten(2) > 0
    gt_mask = targets.flatten(2) > 0
    area_i = torch.sum(pred_mask & gt_mask, dim=-1).float()
    area_u = torch.sum(pred_mask | gt_mask, dim=-1).float()
    actual_ious = area_i / torch.clamp(area_u, min=1.0)

    if use_l1_loss:
        loss = F.l1_loss(pred_ious, actual_ious, reduction="none")
    else:
        loss = F.mse_loss(pred_ious, actual_ious, reduction="none")
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects


class MultiStepMultiMasksAndIous(nn.Module):
    def __init__(
        self,
        weight_dict,
        focal_alpha=0.25,
        focal_gamma=2,
        supervise_all_iou=False,
        iou_use_l1_loss=False,
        pred_obj_scores=False,
        focal_gamma_obj_score=0.0,
        focal_alpha_obj_score=-1,
    ):
        """
        This class computes the multi-step multi-mask and IoU losses.
        Args:
            weight_dict: dict containing weights for focal, dice, iou losses
            focal_alpha: alpha for sigmoid focal loss
            focal_gamma: gamma for sigmoid focal loss
            supervise_all_iou: if True, back-prop iou losses for all predicted masks
            iou_use_l1_loss: use L1 loss instead of MSE loss for iou
            pred_obj_scores: if True, compute loss for object scores
            focal_gamma_obj_score: gamma for sigmoid focal loss on object scores
            focal_alpha_obj_score: alpha for sigmoid focal loss on object scores
        """

        super().__init__()
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        assert "loss_mask" in self.weight_dict
        assert "loss_dice" in self.weight_dict
        assert "loss_iou" in self.weight_dict
        if "loss_class" not in self.weight_dict:
            self.weight_dict["loss_class"] = 0.0

        self.focal_alpha_obj_score = focal_alpha_obj_score
        self.focal_gamma_obj_score = focal_gamma_obj_score
        self.supervise_all_iou = supervise_all_iou
        self.iou_use_l1_loss = iou_use_l1_loss
        self.pred_obj_scores = pred_obj_scores

    def forward(self, outs_batch: List[Dict], targets_batch: torch.Tensor):
        assert len(outs_batch) == len(targets_batch)
        num_objects = torch.tensor(
            (targets_batch.shape[1]), device=targets_batch.device, dtype=torch.float
        )  # Number of objects is fixed within a batch
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_objects)
        num_objects = torch.clamp(num_objects / get_world_size(), min=1).item()

        losses = defaultdict(int)
        for outs, targets in zip(outs_batch, targets_batch):
            cur_losses = self._forward(outs, targets, num_objects)
            for k, v in cur_losses.items():
                losses[k] += v

        return losses

    def _forward(self, outputs: Dict, targets: torch.Tensor, num_objects):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.
        and also the MAE or MSE loss between predicted IoUs and actual IoUs.

        Here "multistep_pred_multimasks_high_res" is a list of multimasks (tensors
        of shape [N, M, H, W], where M could be 1 or larger, corresponding to
        one or multiple predicted masks from a click.

        We back-propagate focal, dice losses only on the prediction channel
        with the lowest focal+dice loss between predicted mask and ground-truth.
        If `supervise_all_iou` is True, we backpropagate ious losses for all predicted masks.
        """

        target_masks = targets.unsqueeze(1).float()
        assert target_masks.dim() == 4  # [N, 1, H, W]
        src_masks_list = outputs["multistep_pred_multimasks_high_res"]
        ious_list = outputs["multistep_pred_ious"]
        object_score_logits_list = outputs["multistep_object_score_logits"]

        assert len(src_masks_list) == len(ious_list)
        assert len(object_score_logits_list) == len(ious_list)

        # accumulate the loss over prediction steps
        losses = {"loss_mask": 0, "loss_dice": 0, "loss_iou": 0, "loss_class": 0}
        for src_masks, ious, object_score_logits in zip(
            src_masks_list, ious_list, object_score_logits_list
        ):
            self._update_losses(
                losses, src_masks, target_masks, ious, num_objects, object_score_logits
            )
        losses[CORE_LOSS_KEY] = self.reduce_loss(losses)
        return losses

    def _update_losses(
        self, losses, src_masks, target_masks, ious, num_objects, object_score_logits
    ):
        target_masks = target_masks.expand_as(src_masks)
        # get focal, dice and iou loss on all output masks in a prediction step
        loss_multimask = sigmoid_focal_loss(
            src_masks,
            target_masks,
            num_objects,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            loss_on_multimask=True,
        )
        loss_multidice = dice_loss(
            src_masks, target_masks, num_objects, loss_on_multimask=True
        )
        if not self.pred_obj_scores:
            loss_class = torch.tensor(
                0.0, dtype=loss_multimask.dtype, device=loss_multimask.device
            )
            target_obj = torch.ones(
                loss_multimask.shape[0],
                1,
                dtype=loss_multimask.dtype,
                device=loss_multimask.device,
            )
        else:
            target_obj = torch.any((target_masks[:, 0] > 0).flatten(1), dim=-1)[
                ..., None
            ].float()
            loss_class = sigmoid_focal_loss(
                object_score_logits,
                target_obj,
                num_objects,
                alpha=self.focal_alpha_obj_score,
                gamma=self.focal_gamma_obj_score,
            )

        loss_multiiou = iou_loss(
            src_masks,
            target_masks,
            ious,
            num_objects,
            loss_on_multimask=True,
            use_l1_loss=self.iou_use_l1_loss,
        )
        assert loss_multimask.dim() == 2
        assert loss_multidice.dim() == 2
        assert loss_multiiou.dim() == 2
        if loss_multimask.size(1) > 1:
            # take the mask indices with the smallest focal + dice loss for back propagation
            loss_combo = (
                loss_multimask * self.weight_dict["loss_mask"]
                + loss_multidice * self.weight_dict["loss_dice"]
            )
            best_loss_inds = torch.argmin(loss_combo, dim=-1)
            batch_inds = torch.arange(loss_combo.size(0), device=loss_combo.device)
            loss_mask = loss_multimask[batch_inds, best_loss_inds].unsqueeze(1)
            loss_dice = loss_multidice[batch_inds, best_loss_inds].unsqueeze(1)
            # calculate the iou prediction and slot losses only in the index
            # with the minimum loss for each mask (to be consistent w/ SAM)
            if self.supervise_all_iou:
                loss_iou = loss_multiiou.mean(dim=-1).unsqueeze(1)
            else:
                loss_iou = loss_multiiou[batch_inds, best_loss_inds].unsqueeze(1)
        else:
            loss_mask = loss_multimask
            loss_dice = loss_multidice
            loss_iou = loss_multiiou

        # backprop focal, dice and iou loss only if obj present
        loss_mask = loss_mask * target_obj
        loss_dice = loss_dice * target_obj
        loss_iou = loss_iou * target_obj

        # sum over batch dimension (note that the losses are already divided by num_objects)
        losses["loss_mask"] += loss_mask.sum()
        losses["loss_dice"] += loss_dice.sum()
        losses["loss_iou"] += loss_iou.sum()
        losses["loss_class"] += loss_class

    def reduce_loss(self, losses):
        reduced_loss = 0.0
        for loss_key, weight in self.weight_dict.items():
            if loss_key not in losses:
                raise ValueError(f"{type(self)} doesn't compute {loss_key}")
            if weight != 0:
                reduced_loss += losses[loss_key] * weight

        return reduced_loss


# Point-based loss functions
def _ensure_nchw(x):
    """Ensure tensor is in NCHW format"""
    if x.dim() == 3:
        x = x.unsqueeze(1)
    return x


@torch.no_grad()
def sample_points_by_uncertainty(
    logits, num_points: int = 2048, oversample_ratio: float = 3.0, importance_sample_ratio: float = 0.75,
):
    """
    Sample points based on uncertainty (similar to Detectron2's get_uncertain_point_coords_with_randomness)
    Args:
        logits: [N, C, H, W] tensor of logits
        num_points: number of points to sample
        oversample_ratio: oversample ratio for importance sampling
        importance_sample_ratio: ratio of points to sample from high uncertainty regions
    Returns:
        y_idx, x_idx: [N, K] tensors of y and x coordinates
    """
    assert logits.dim() == 4, "logits must be [N,C,H,W]"
    N, C, H, W = logits.shape
    K = int(num_points)
    K_over = max(int(K * oversample_ratio), K)
    K_imp = min(K, int(K * importance_sample_ratio))
    device = logits.device

    with torch.no_grad():
        # Random sampling
        y_rand = torch.randint(0, H, size=(N, K_over), device=device)
        x_rand = torch.randint(0, W, size=(N, K_over), device=device)
        inds = y_rand * W + x_rand
        flat = logits.view(N, C, H * W)
        cand_logits = torch.gather(flat, dim=2, index=inds.unsqueeze(1).expand(N, C, K_over))
        
        # Calculate uncertainty (negative absolute value of logits)
        uncertainty = -torch.abs(cand_logits).mean(dim=1)

        # Importance sampling from high uncertainty regions
        if K_imp > 0:
            topk_vals, topk_idx = torch.topk(uncertainty, k=K_imp, dim=1, largest=True, sorted=False)
            y_top = torch.gather(y_rand, 1, topk_idx)
            x_top = torch.gather(x_rand, 1, topk_idx)
        else:
            y_top = torch.empty((N, 0), dtype=torch.long, device=device)
            x_top = torch.empty((N, 0), dtype=torch.long, device=device)

        # Random sampling for remaining points
        K_rem = K - K_imp
        if K_rem > 0:
            y_rest = torch.randint(0, H, size=(N, K_rem), device=device)
            x_rest = torch.randint(0, W, size=(N, K_rem), device=device)
            y_idx = torch.cat([y_top, y_rest], dim=1)
            x_idx = torch.cat([x_top, x_rest], dim=1)
        else:
            y_idx, x_idx = y_top, x_top

    return y_idx, x_idx


def _gather_points_from_maps(maps, y_idx, x_idx):
    """
    Gather point values from 2D maps
    Args:
        maps: [N, C, H, W] tensor
        y_idx, x_idx: [N, K] tensors of coordinates
    Returns:
        sampled: [N, C, K] tensor of sampled values
    """
    N, C, H, W = maps.shape
    K = y_idx.shape[1]
    lin = y_idx * W + x_idx
    flat = maps.view(N, C, H * W)
    sampled = torch.gather(flat, dim=2, index=lin.unsqueeze(1).expand(N, C, K))
    return sampled


def point_bce_loss(
    logits, targets, num_objects: float,
    num_points: int = 2048, oversample_ratio: float = 3.0, importance_sample_ratio: float = 0.75,
):
    """
    Point-based BCE loss using uncertainty-based sampling
    Args:
        logits: [N, C, H, W] tensor of logits
        targets: [N, C, H, W] tensor of targets
        num_objects: number of objects in batch
        num_points: number of points to sample
        oversample_ratio: oversample ratio for importance sampling
        importance_sample_ratio: ratio of points to sample from high uncertainty regions
    Returns:
        loss: scalar loss value
    """
    logits = _ensure_nchw(logits).float()
    targets = _ensure_nchw(targets).float()
    assert logits.shape == targets.shape, "logits/targets shape mismatch"

    N, C, H, W = logits.shape
    y_idx, x_idx = sample_points_by_uncertainty(
        logits=logits, num_points=num_points,
        oversample_ratio=oversample_ratio, importance_sample_ratio=importance_sample_ratio,
    )
    logits_pts = _gather_points_from_maps(logits, y_idx, x_idx)
    targets_pts = _gather_points_from_maps(targets, y_idx, x_idx)

    loss = F.binary_cross_entropy_with_logits(logits_pts, targets_pts, reduction="none")
    loss = loss.mean(dim=2).sum() / max(num_objects, 1.0)
    return loss


def point_dice_loss(
    logits, targets, num_objects: float,
    num_points: int = 2048, oversample_ratio: float = 3.0, importance_sample_ratio: float = 0.75,
    eps: float = 1e-7,
):
    """
    Point-based Dice loss using uncertainty-based sampling
    Args:
        logits: [N, C, H, W] tensor of logits
        targets: [N, C, H, W] tensor of targets
        num_objects: number of objects in batch
        num_points: number of points to sample
        oversample_ratio: oversample ratio for importance sampling
        importance_sample_ratio: ratio of points to sample from high uncertainty regions
        eps: epsilon for numerical stability
    Returns:
        loss: scalar loss value
    """
    logits = _ensure_nchw(logits).float()
    targets = _ensure_nchw(targets).float()
    assert logits.shape == targets.shape, "logits/targets shape mismatch"

    N, C, H, W = logits.shape
    y_idx, x_idx = sample_points_by_uncertainty(
        logits=logits, num_points=num_points,
        oversample_ratio=oversample_ratio, importance_sample_ratio=importance_sample_ratio,
    )
    prob_pts = torch.sigmoid(_gather_points_from_maps(logits, y_idx, x_idx))
    tgt_pts = _gather_points_from_maps(targets, y_idx, x_idx)

    inter = (prob_pts * tgt_pts).sum(dim=2)
    denom = prob_pts.sum(dim=2) + tgt_pts.sum(dim=2)
    dice = 1.0 - (2.0 * inter + 1.0) / (denom + 1.0 + eps)
    loss = dice.sum() / max(num_objects, 1.0)
    return loss


class MultiStepMultiPointsAndIous(nn.Module):
    """
    Multi-step multi-mask and IoU losses using point-based sampling
    This class is similar to MultiStepMultiMasksAndIous but uses point-based BCE and Dice losses
    instead of full-mask losses for better efficiency and focus on uncertain regions.
    """
    def __init__(
        self,
        weight_dict,
        focal_alpha=0.25,
        focal_gamma=2,
        supervise_all_iou=False,
        iou_use_l1_loss=False,
        pred_obj_scores=False,
        focal_gamma_obj_score=0.0,
        focal_alpha_obj_score=-1,
        num_points=2048,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
    ):
        """
        Args:
            weight_dict: dict containing weights for focal, dice, iou losses
            focal_alpha: alpha for sigmoid focal loss
            focal_gamma: gamma for sigmoid focal loss
            supervise_all_iou: if True, back-prop iou losses for all predicted masks
            iou_use_l1_loss: use L1 loss instead of MSE loss for iou
            pred_obj_scores: if True, compute loss for object scores
            focal_gamma_obj_score: gamma for sigmoid focal loss on object scores
            focal_alpha_obj_score: alpha for sigmoid focal loss on object scores
            num_points: number of points to sample for point-based losses
            oversample_ratio: oversample ratio for importance sampling
            importance_sample_ratio: ratio of points to sample from high uncertainty regions
        """
        super().__init__()
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        assert "loss_mask" in self.weight_dict
        assert "loss_dice" in self.weight_dict
        assert "loss_iou" in self.weight_dict
        if "loss_class" not in self.weight_dict:
            self.weight_dict["loss_class"] = 0.0

        self.focal_alpha_obj_score = focal_alpha_obj_score
        self.focal_gamma_obj_score = focal_gamma_obj_score
        self.supervise_all_iou = supervise_all_iou
        self.iou_use_l1_loss = iou_use_l1_loss
        self.pred_obj_scores = pred_obj_scores
        
        # Point sampling parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def forward(self, outs_batch: List[Dict], targets_batch: torch.Tensor):
        assert len(outs_batch) == len(targets_batch)
        num_objects = torch.tensor(
            (targets_batch.shape[1]), device=targets_batch.device, dtype=torch.float
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_objects)
        num_objects = torch.clamp(num_objects / get_world_size(), min=1).item()

        losses = defaultdict(int)
        for outs, targets in zip(outs_batch, targets_batch):
            cur_losses = self._forward(outs, targets, num_objects)
            for k, v in cur_losses.items():
                losses[k] += v

        return losses

    def _forward(self, outputs: Dict, targets: torch.Tensor, num_objects):
        """
        Compute point-based losses for masks and IoU losses
        """
        target_masks = targets.unsqueeze(1).float()
        assert target_masks.dim() == 4  # [N, 1, H, W]
        src_masks_list = outputs["multistep_pred_multimasks_high_res"]
        ious_list = outputs["multistep_pred_ious"]
        object_score_logits_list = outputs["multistep_object_score_logits"]

        assert len(src_masks_list) == len(ious_list)
        assert len(object_score_logits_list) == len(ious_list)

        # accumulate the loss over prediction steps
        losses = {"loss_mask": 0, "loss_dice": 0, "loss_iou": 0, "loss_class": 0}
        for src_masks, ious, object_score_logits in zip(
            src_masks_list, ious_list, object_score_logits_list
        ):
            self._update_losses(
                losses, src_masks, target_masks, ious, num_objects, object_score_logits
            )
        losses[CORE_LOSS_KEY] = self.reduce_loss(losses)
        return losses

    def _update_losses(
        self, losses, src_masks, target_masks, ious, num_objects, object_score_logits
    ):
        target_masks = target_masks.expand_as(src_masks)
        
        # Handle object scores
        if not self.pred_obj_scores:
            loss_class = torch.tensor(
                0.0, dtype=src_masks.dtype, device=src_masks.device
            )
            target_obj = torch.ones(
                src_masks.shape[0],
                1,
                dtype=src_masks.dtype,
                device=src_masks.device,
            )
        else:
            target_obj = torch.any((target_masks[:, 0] > 0).flatten(1), dim=-1)[
                ..., None
            ].float()
            loss_class = sigmoid_focal_loss(
                object_score_logits,
                target_obj,
                num_objects,
                alpha=self.focal_alpha_obj_score,
                gamma=self.focal_gamma_obj_score,
            )

        # Compute point-based losses for each mask
        N, M, H, W = src_masks.shape
        loss_multimask = torch.zeros(N, M, device=src_masks.device)
        loss_multidice = torch.zeros(N, M, device=src_masks.device)
        
        for m in range(M):
            # Sample points based on uncertainty for current mask
            y_idx, x_idx = sample_points_by_uncertainty(
                src_masks[:, m:m+1], 
                num_points=self.num_points,
                oversample_ratio=self.oversample_ratio, 
                importance_sample_ratio=self.importance_sample_ratio,
            )
            
            # Gather point values
            logits_pts = _gather_points_from_maps(src_masks[:, m:m+1], y_idx, x_idx)
            targets_pts = _gather_points_from_maps(target_masks[:, m:m+1], y_idx, x_idx)
            
            # Compute point-based BCE loss
            bce_loss = F.binary_cross_entropy_with_logits(logits_pts, targets_pts, reduction="none")
            loss_multimask[:, m] = bce_loss.mean(dim=2).mean(dim=1)
            
            # Compute point-based Dice loss
            prob_pts = torch.sigmoid(logits_pts)
            inter = (prob_pts * targets_pts).sum(dim=2)
            denom = prob_pts.sum(dim=2) + targets_pts.sum(dim=2)
            dice = 1.0 - (2.0 * inter + 1.0) / (denom + 1.0 + 1e-7)
            loss_multidice[:, m] = dice.mean(dim=1)

        # Normalize by number of objects
        loss_multimask = loss_multimask / max(num_objects, 1.0)
        loss_multidice = loss_multidice / max(num_objects, 1.0)

        # Compute IoU loss (same as original)
        loss_multiiou = iou_loss(
            src_masks,
            target_masks,
            ious,
            num_objects,
            loss_on_multimask=True,
            use_l1_loss=self.iou_use_l1_loss,
        )

        # Select best mask based on combined loss
        if M > 1:
            loss_combo = (
                loss_multimask * self.weight_dict["loss_mask"]
                + loss_multidice * self.weight_dict["loss_dice"]
            )
            best_loss_inds = torch.argmin(loss_combo, dim=-1)
            batch_inds = torch.arange(loss_combo.size(0), device=loss_combo.device)
            loss_mask = loss_multimask[batch_inds, best_loss_inds].unsqueeze(1)
            loss_dice = loss_multidice[batch_inds, best_loss_inds].unsqueeze(1)
            
            if self.supervise_all_iou:
                loss_iou = loss_multiiou.mean(dim=-1).unsqueeze(1)
            else:
                loss_iou = loss_multiiou[batch_inds, best_loss_inds].unsqueeze(1)
        else:
            loss_mask = loss_multimask
            loss_dice = loss_multidice
            loss_iou = loss_multiiou

        # Apply object presence mask
        loss_mask = loss_mask * target_obj
        loss_dice = loss_dice * target_obj
        loss_iou = loss_iou * target_obj

        # Sum over batch dimension
        losses["loss_mask"] += loss_mask.sum()
        losses["loss_dice"] += loss_dice.sum()
        losses["loss_iou"] += loss_iou.sum()
        losses["loss_class"] += loss_class

    def reduce_loss(self, losses):
        reduced_loss = 0.0
        for loss_key, weight in self.weight_dict.items():
            if loss_key not in losses:
                raise ValueError(f"{type(self)} doesn't compute {loss_key}")
            if weight != 0:
                reduced_loss += losses[loss_key] * weight

        return reduced_loss
