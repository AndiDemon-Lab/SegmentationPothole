import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

def dice_score(pred, target):
    pred = pred.astype(np.uint8)
    target = target.astype(np.uint8)
    inter = (pred & target).sum()
    return (2 * inter + 1e-6) / (pred.sum() + target.sum() + 1e-6)

def iou_score(pred, target):
    inter = (pred & target).sum()
    union = (pred | target).sum()
    return (inter + 1e-6) / (union + 1e-6)

def precision_recall_f1(pred, target):
    p = precision_score(target, pred, zero_division=0)
    r = recall_score(target, pred, zero_division=0)
    f = f1_score(target, pred, zero_division=0)
    return p, r, f

def plot_loss(losses, save_path):
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid()
    plt.savefig(save_path)
    plt.close()
