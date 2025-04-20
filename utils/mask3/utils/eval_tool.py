import torch
import numpy as np


def calculate_iou(pred_mask, true_mask):
    """
    Calculate IoU (Intersection over Union) for binary masks.

    Parameters:
        pred_mask: numpy array, predicted mask (binary).
        true_mask: numpy array, ground truth mask (binary).

    Returns:
        iou: IoU value.
    """
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    iou = intersection / (union + 1e-6)  # Avoid division by zero
    return iou


def calculate_tp_fp_fn(pred_mask, true_mask, iou_threshold):
    """
    Calculate TP, FP, FN based on IoU threshold.

    Parameters:
        pred_mask: numpy array, predicted mask (binary).
        true_mask: numpy array, ground truth mask (binary).
        iou_threshold: float, IoU threshold for determining TP/FP.

    Returns:
        TP: True Positive count.
        FP: False Positive count.
        FN: False Negative count.
    """
    iou = calculate_iou(pred_mask, true_mask)
    if iou >= iou_threshold:
        TP = 1
        FP = 0
        FN = 0
    else:
        TP = 0
        FP = 1 if pred_mask.sum() > 0 else 0
        FN = 1 if true_mask.sum() > 0 else 0
    return TP, FP, FN


def calculate_map(pred_masks, true_masks, iou_thresholds=np.arange(0.5, 1.0, 0.05)):
    """
    Calculate mAP@50-95 for multiple masks.

    Parameters:
        pred_masks: numpy array, predicted masks (binary), shape (N, H, W).
        true_masks: numpy array, ground truth masks (binary), shape (N, H, W).
        iou_thresholds: array of IoU thresholds (default: 0.5 to 0.95 with step 0.05).

    Returns:
        mAP: Mean Average Precision across IoU thresholds.
    """
    aps = []
    for threshold in iou_thresholds:
        tp_list, fp_list, fn_list = [], [], []
        for pred_mask, true_mask in zip(pred_masks, true_masks):
            TP, FP, FN = calculate_tp_fp_fn(pred_mask, true_mask, threshold)
            tp_list.append(TP)
            fp_list.append(FP)
            fn_list.append(FN)

        # Calculate precision and recall
        TP = sum(tp_list)
        FP = sum(fp_list)
        FN = sum(fn_list)
        precision = TP / (TP + FP + 1e-6)
        recall = TP / (TP + FN + 1e-6)

        # PR curve interpolation
        recall_points = np.linspace(0, 1, 101)
        precisions = []
        for recall_point in recall_points:
            if recall >= recall_point:
                precisions.append(precision)
            else:
                precisions.append(0)
        ap = np.mean(precisions)
        aps.append(ap)

    # Mean Average Precision (mAP)
    mAP = np.mean(aps)
    return mAP



#得到混淆矩阵
def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

#计算图像分割衡量系数
def label_accuracy_score(label_trues, label_preds, n_class):
    """
     :param label_preds: numpy data, shape:[batch,h,w]
     :param label_trues:同上
     :param n_class:类别数
     Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues,label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()

    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)

    iu = np.diag(hist) / ( hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) )
    mean_iu = np.nanmean(iu)

    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc

if __name__ == "__main__":
    # 假设 pred_masks 和 true_masks 是 2x640x640 的二值化矩阵
    pred_masks = np.random.randint(1, 8, (2, 640, 640))  # 随机预测 mask
    true_masks = np.random.randint(1, 8, (2, 640, 640))  # 随机真实 mask

    # 计算 mAP@50-95
    map50_95 = calculate_map(pred_masks, true_masks)
    print(f"mAP@50-95: {map50_95:.4f}")