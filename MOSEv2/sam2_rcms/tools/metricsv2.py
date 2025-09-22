import math
import numpy as np
import cv2

from skimage.morphology import disk


def db_eval_iou(annotation, segmentation):
    """ Compute region similarity as the Jaccard Index.
    Arguments:
        annotation   (ndarray): binary annotation   map.
        segmentation (ndarray): binary segmentation map.
        void_pixels  (ndarray): optional mask with void pixels

    Return:
        jaccard (float): region similarity
    """
    assert annotation.shape == segmentation.shape, \
        f'Annotation({annotation.shape}) and segmentation:{segmentation.shape} dimensions do not match.'

    # Intersection between all sets
    inters = np.sum(np.logical_and(segmentation, annotation), axis=(-2, -1))
    union = np.sum(np.logical_or(segmentation, annotation), axis=(-2, -1))

    j = (inters + 1) / (union + 1)
    j = np.clip(j, 0, 1)
    if j.ndim == 0:
        j = 1 if np.isclose(union, 0) else j
    else:
        j[np.isclose(union, 0)] = 1
    empty_gt = np.sum(annotation, axis=(-2, -1)) == 0
    return j, empty_gt


def db_eval_boundary(annotation, segmentation, bound_th=0.008, bound_mode=None):
    assert annotation.shape == segmentation.shape
    if annotation.ndim == 3:
        n_frames = annotation.shape[0]
        f_res = np.zeros(n_frames)
        f_res_ = np.zeros(n_frames)
        for frame_id in range(n_frames):
            if bound_mode is None:
                f_res[frame_id], f_res_[frame_id] = f_measure(segmentation[frame_id, :, :, ], annotation[frame_id, :, :])
            else:
                f_res[frame_id], f_res_[frame_id] = f_measure_adaptive(segmentation[frame_id, :, :, ], annotation[frame_id, :, :], bound_th=bound_th, mode=bound_mode)
    elif annotation.ndim == 2:
        if bound_mode is None:
            f_res, f_res_ = f_measure(segmentation, annotation)
        else:
            f_res, f_res_ = f_measure_adaptive(segmentation, annotation, bound_th=bound_th, mode=bound_mode)
    else:
        raise ValueError(f'db_eval_boundary does not support tensors with {annotation.ndim} dimensions')
    return f_res, f_res_


def f_measure(foreground_mask, gt_mask, bound_th=0.008):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.

    Arguments:
        foreground_mask (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.
        void_pixels     (ndarray): optional mask with void pixels

    Returns:
        F (float): boundaries F-measure
    """
    assert np.atleast_3d(foreground_mask).shape[2] == 1

    bound_pix = bound_th if bound_th >= 1 else \
        np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))

    # Get the pixel boundaries of both masks
    foreground_mask[foreground_mask > 0] = 1
    gt_mask[gt_mask > 0] = 1
    
    fg_boundary = _seg2bmap(foreground_mask)
    gt_boundary = _seg2bmap(gt_mask)
    
    fg_dil = cv2.dilate(fg_boundary, disk(bound_pix))
    gt_dil = cv2.dilate(gt_boundary, disk(bound_pix))
    
    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil
    
    # Area of the intersection
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    # % Compute precision and recall
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)
    
    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall)
    
    return F


def _seg2bmap(seg, width=None, height=None):

    # seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (
        width > w | height > h | abs(ar1 - ar2) > 0.01
    ), "Can" "t convert %dx%d seg to %dx%d bmap." % (w, h, width, height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + math.floor((y - 1) + height / h)
                    i = 1 + math.floor((x - 1) + width / h)
                    bmap[j, i] = 1

    return bmap


def f_measure_adaptive(foreground_mask, gt_mask, bound_th=0.008, mode='diagonal'):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.

    Arguments:
        foreground_mask (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.
        void_pixels     (ndarray): optional mask with void pixels
        mode            (str): 'diagonal' or 'area' to determine how to calculate boundary width

    Returns:
        F (float): boundaries F-measure
    """
    assert np.atleast_3d(foreground_mask).shape[2] == 1
    assert mode in ['diagonal', 'area'], "Mode must be either 'diagonal' or 'area'"
    
    # Calculate the boundary pixel width based on the selected mode
    if bound_th >= 1:
        bound_pix = bound_th
    else:
        # Find bounding box of non-zero elements in gt_mask
        if np.any(gt_mask):
            rows = np.any(gt_mask, axis=1)
            cols = np.any(gt_mask, axis=0)
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            
            if mode == 'diagonal':
                # Calculate diagonal length of the bounding box
                diagonal_length = np.sqrt((y_max - y_min)**2 + (x_max - x_min)**2)
                bound_pix = np.ceil(bound_th * diagonal_length)
            else:  # mode == 'area'
                # Calculate based on the area of the object
                area = np.sum(gt_mask > 0)
                bound_pix = np.ceil(bound_th * np.sqrt(area))
        else:
            # Fallback to original calculation if gt_mask is empty
            bound_pix = np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))
        
        bound_pix = min(0.008 *  np.linalg.norm(foreground_mask.shape), bound_pix)
    
    if bound_pix < 0.008 *  np.linalg.norm(foreground_mask.shape):
        f = f_measure_func(foreground_mask, gt_mask, bound_th=bound_th, bound_pix=np.ceil(0.008 * np.linalg.norm(foreground_mask.shape)))
        f_ = f_measure_func(foreground_mask, gt_mask, bound_th=bound_th, bound_pix=bound_pix)
    else:
        f = f_measure_func(foreground_mask, gt_mask, bound_th=bound_th, bound_pix=0.008 *  np.linalg.norm(foreground_mask.shape))
        f_ = f
    return f, f_


def f_measure_func(foreground_mask, gt_mask, bound_th=0.008, bound_pix=3):
    # Get the pixel boundaries of both masks
    foreground_mask[foreground_mask > 0] = 1
    gt_mask[gt_mask > 0] = 1
    fg_boundary = _seg2bmap(foreground_mask)
    gt_boundary = _seg2bmap(gt_mask)
    
    fg_dil = cv2.dilate(fg_boundary, disk(bound_pix))
    gt_dil = cv2.dilate(gt_boundary, disk(bound_pix))
    
    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)
    
    # % Compute precision and recall
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall)
    
    return F


if __name__ == '__main__':
    pass
