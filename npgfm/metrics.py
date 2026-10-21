"""Image- and pixel-level anomaly-detection metrics."""
import numpy as np
from scipy.ndimage import label
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc


def f1_max(y, s):
    p, r, _ = precision_recall_curve(y, s)
    return float(np.max(2 * p * r / np.clip(p + r, 1e-12, None)))


def aupro(masks, amaps, num_th=200, max_fpr=0.3):
    """Area under the per-region-overlap curve up to `max_fpr`, normalised to [0, 1]."""
    regions = []  # (image index, flat pixel indices) of every connected GT region
    for i, m in enumerate(masks):
        lab, n = label(m, structure=np.ones((3, 3)))  # 8-connected regions
        regions += [(i, np.flatnonzero(lab == k)) for k in range(1, n + 1)]
    normal = masks == 0
    pros, fprs = [], []
    for th in np.linspace(amaps.min(), amaps.max(), num_th, endpoint=False):
        b = amaps > th
        pros.append(np.mean([b[i].ravel()[idx].mean() for i, idx in regions]))
        fprs.append(np.logical_and(b, normal).sum() / normal.sum())
    pros, fprs = np.array(pros), np.array(fprs)
    keep = fprs < max_fpr
    return float(auc(fprs[keep] / fprs[keep].max(), pros[keep]))


def compute_metrics(masks, amaps):
    """masks: (N,H,W) binary ground truth; amaps: (N,H,W) anomaly maps. Image score = max of the map."""
    gt_img, sc_img = masks.reshape(len(masks), -1).max(1), amaps.reshape(len(amaps), -1).max(1)
    gt_pix, sc_pix = masks.ravel().astype(int), amaps.ravel()
    return {
        'I-AUROC': roc_auc_score(gt_img, sc_img),
        'I-AP': average_precision_score(gt_img, sc_img),
        'I-F1max': f1_max(gt_img, sc_img),
        'P-AUROC': roc_auc_score(gt_pix, sc_pix),
        'P-AP': average_precision_score(gt_pix, sc_pix),
        'P-F1max': f1_max(gt_pix, sc_pix),
        'AUPRO': aupro(masks, amaps),
    }
