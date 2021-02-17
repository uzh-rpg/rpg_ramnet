from sklearn.metrics import mean_squared_error
from skimage.measure import compare_ssim as ssim
import torch
import numpy as np
from scipy import ndimage


def abs_rel_diff(y_input, y_target, eps = 1e-6):
    abs_diff = np.abs(y_target-y_input)
    return (abs_diff[~np.isnan(abs_diff)]/(y_target[~np.isnan(y_target)]+eps)).mean()

def squ_rel_diff(y_input, y_target, eps = 1e-6):
    abs_diff = np.abs(y_target-y_input)
    is_nan = np.isnan(abs_diff)
    return (abs_diff[~is_nan]**2/(y_target[~is_nan]**2+eps)).mean()

def rms_linear(y_input, y_target):
    abs_diff = np.abs(y_target-y_input)
    is_nan = np.isnan(abs_diff)
    return np.sqrt((abs_diff[~is_nan]**2).mean())

def scale_invariant_error(y_input, y_target):
    log_diff = np.abs(y_target-y_input)
    is_nan = np.isnan(log_diff)
    return (log_diff[~is_nan]**2).mean()-(log_diff[~is_nan].mean())**2

def mean_error(y_input, y_target):
    abs_diff = np.abs(y_target-y_input)
    return abs_diff[~np.isnan(abs_diff)].mean()

def median_error(y_input, y_target):
    abs_diff = np.abs(y_target-y_input)
    return np.median(abs_diff[~np.isnan(abs_diff)])

def mse(y_input, y_target):
    N, C, H, W = y_input.shape
    assert(C == 1 or C == 3)
    sum_mse_over_batch = 0.

    for i in range(N):
        sum_mse_over_batch += mean_squared_error(
            y_input[i, 0, :, :][~np.isnan(y_target[i, 0, :, :])], y_target[i, 0, :, :][~np.isnan(y_target[i, 0, :, :])])

        if C == 3:  # color
            sum_mse_over_batch += mean_squared_error(
                y_input[i, 1, :, :][~np.isnan(y_target[i, 1, :, :])], y_target[i, 1, :, :][~np.isnan(y_target[i, 1, :, :])])
            sum_mse_over_batch += mean_squared_error(
                y_input[i, 2, :, :][~np.isnan(y_target[i, 2, :, :])], y_target[i, 2, :, :][~np.isnan(y_target[i, 2, :, :])])

    mean_mse = sum_mse_over_batch / (float(N))
    if C == 3:
        mean_mse /= 3.0

    return mean_mse


def structural_similarity(y_input, y_target):
    N, C, H, W = y_input.shape
    assert(C == 1 or C == 3)
    # N x C x H x W -> N x W x H x C -> N x H x W x C
    y_input = np.swapaxes(y_input, 1, 3)
    y_input = np.swapaxes(y_input, 1, 2)
    y_target = np.swapaxes(y_target, 1, 3)
    y_target = np.swapaxes(y_target, 1, 2)
    sum_structural_similarity_over_batch = 0.
    for i in range(N):
        if C == 3:
            sum_structural_similarity_over_batch += ssim(
                y_input[i, :, :, :], y_target[i, :, :, :], multichannel=True)
        else:
            sum_structural_similarity_over_batch += ssim(
                y_input[i, :, :, 0], y_target[i, :, :, 0])

    return sum_structural_similarity_over_batch / float(N)
