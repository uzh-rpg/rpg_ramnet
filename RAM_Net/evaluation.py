import numpy as np
import torch
import argparse
import glob
from os.path import join
import tqdm
import cv2
import torch.nn.functional as f

from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from model.metric import *


def FLAGS():
    parser = argparse.ArgumentParser("""Event Depth Data estimation.""")

    # training / validation dataset
    parser.add_argument("--target_dataset", default="", required=True)
    parser.add_argument("--predictions_dataset", default="", required=True)
    parser.add_argument("--event_masks", default="")
    parser.add_argument("--crop_ymax", default=260, type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--idx", type=int, default=-1)
    parser.add_argument("--start_idx", type=int, default=-1)
    parser.add_argument("--prediction_offset", type=int, default=0)
    parser.add_argument("--target_offset", type=int, default=0)
    parser.add_argument("--rescale", action="store_true", default=False)
    parser.add_argument("--clip_distance", type=float, default=80.0)
    parser.add_argument("--output_folder", type=str, default=None)
    parser.add_argument("--down_scale_factor", type=float, default=1.0)
    parser.add_argument("--reg_factor", type=float, default=0.0)

    flags = parser.parse_args()
    return flags


depth_values = [10, 20, 30, 80, 250, 500]

metrics_keywords = [f"_abs_rel_diff",
                    f"_squ_rel_diff",
                    f"_RMS_linear",
                    f"_RMS_log",
                    f"_SILog",
                    f"_mean_depth_error",
                    f"_median_diff",
                    f"_threshold_delta_1.25",
                    f"_threshold_delta_1.25^2",
                    f"_threshold_delta_1.25^3"]

for k in depth_values:
    metrics_keywords.append(f"_{k}_abs_rel_diff")
    metrics_keywords.append(f"_{k}_squ_rel_diff")
    metrics_keywords.append(f"_{k}_RMS_linear")
    metrics_keywords.append(f"_{k}_RMS_log")
    metrics_keywords.append(f"_{k}_SILog")
    metrics_keywords.append(f"_{k}_mean_depth_error")
    metrics_keywords.append(f"_{k}_median_diff")
    metrics_keywords.append(f"_{k}_threshold_delta_1.25")
    metrics_keywords.append(f"_{k}_threshold_delta_1.25^2")
    metrics_keywords.append(f"_{k}_threshold_delta_1.25^3")

def eval_metrics(output, target):
    metrics = [mse, abs_rel_diff, scale_invariant_error, median_error, mean_error, rms_linear]
    acc_metrics = np.zeros(len(metrics))
    output = output[None, :][None, :]
    target = target[None, :][None, :]
    for i, metric in enumerate(metrics):
        acc_metrics[i] += metric(output, target)
    return acc_metrics


def prepare_depth_data(target, prediction, clip_distance, down_scale_factor=1.0, reg_factor=0.0):
    # retreiv metric depth from log depth
    prediction = np.exp(reg_factor * (prediction - np.ones((prediction.shape[0], prediction.shape[1]), dtype=np.float32)))
    target = np.exp(reg_factor * (target - np.ones((target.shape[0], target.shape[1]), dtype=np.float32)))

    # Get back to the absolute values
    target *= clip_distance
    prediction *= clip_distance

    scale = 1.0
    prediction = prediction / scale
    prediction = np.clip(prediction, np.exp(-1 * reg_factor) * clip_distance, clip_distance)

    if down_scale_factor < 1.0:
        target = torch.tensor(target)
        target = target[None]
        target = target[None]
        target = f.interpolate(target, scale_factor=down_scale_factor, mode='bilinear')
        target = target[0]
        target = target[0]
        return target.numpy(), prediction

    return target, prediction


def rescale_by_the_median(target, prediction, debug = False):
    if debug:
        print("target median:", np.median(target))
        print("target std:", np.std(target))

    target = (target - np.median(target))/np.std(target)
    target = target + abs(np.min(target))

    if debug:
        print("target median[scaled]:", np.median(target))

    if debug:
        print("prediction median:", np.median(prediction))
        print("prediction std:", np.std(prediction))

    prediction = (prediction - np.median(prediction))/np.std(prediction)
    prediction = prediction + abs(np.min(prediction))

    if debug:
        print("prediction median[scaled]:", np.median(prediction))

    # Adjust by the median
    median_diff = np.abs(np.median(target) - np.median(prediction))
    if np.median(target) < np.median(prediction):
        target += median_diff
    else:
        prediction += median_diff

    #target *= 1000.00 
    #prediction *= 1000.00
 
    if debug:
        print("target median[adjusted]:", np.median(target))
        print("prediction median[adjusted]:", np.median(prediction))
        print("target min[adjusted]:", np.min(target))
        print("target max[adjusted]:", np.max(target))
        print("prediction min[adjusted]:", np.min(prediction))
        print("prediction max[adjusted]:", np.max(prediction))
        
    #if np.min(target) < 0:
    #    target += np.min(target)
    #elif np.min(target) > 0:
    #    target -= np.min(target)
    #if np.min(prediction) < 0:
    #    prediction += np.min(prediction)
    #elif np.min(prediction) > 0:
    #    prediction += np.min(prediction)

    
    #if debug:
    #    print("target max[adjusted]:", np.max(target))
    #    print("prediction max[adjusted]:", np.max(prediction))
    #    print("target min[adjusted]:", np.min(target))
    #    print("prediction min[adjusted]:", np.min(prediction))

    return target, prediction

def display_high_contrast_colormap (idx, target, prediction, prefix="", colormap = 'terrain', debug=False, folder_name=None):

    if folder_name is not None or debug:
        percent = 1.0
        second_largest = sorted(list(set(target.flatten().tolist())))[-2]
        fig, ax = plt.subplots(ncols=1, nrows=2)
        target_plot = np.flip(np.fliplr(np.clip(target, 0, percent*np.max(target))))
        #ax[0].contour(target_plot, levels=[0.5 * np.median(target)], colors='k', linestyles='-')
        #pcm = ax[0].pcolormesh(target_plot, cmap=colormap, vmin=np.min(target), vmax = percent * np.max(target))
        pcm = ax[0].pcolormesh(target_plot, cmap=colormap, vmin=np.min(target), vmax=percent*second_largest)
        ax[0].set_xticklabels([]) # no tick numbers in the target plot horizontal axis
        ax[0].set_title("Target")
        fig.colorbar(pcm, ax=ax[0], extend='both', orientation='vertical')
        prediction_plot = np.flip(np.fliplr(np.clip(prediction, 0, percent*np.max(prediction))))
        #ax[1].contour(prediction_plot, levels=[0.5 * np.median(target)], colors='k', linestyles='-')
        #pcm = ax[1].pcolormesh(prediction_plot, cmap=colormap, vmin=np.min(target), vmax = percent * np.max(target))
        pcm = ax[1].pcolormesh(prediction_plot, cmap=colormap, vmin=np.min(target), vmax=percent*second_largest)
        ax[1].set_title("Prediction")
        fig.colorbar(pcm, ax=ax[1], extend='both', orientation='vertical')
        fig.canvas.set_window_title(prefix+"High_Contrast_Depth_Evaluation")
    if folder_name is not None:
        plt.savefig('%s/frame_%010d.png' % (folder_name, idx))
        plt.close(fig)
    if debug:
        plt.show()

def display_high_contrast_color_logmap (idx, data, prefix="", name="data", colormap = 'tab20c', debug=False, folder_name=None):

    if debug and folder_name is not None:
        percent = 1.0
        fig, ax = plt.subplots(ncols=1, nrows=1)
        target_plot = np.flip(np.fliplr(np.clip(data, 0, percent*np.max(data))))
        #print ("median: ", np.median(data))
        #ax.contour(target_plot, Z = np.median(data), levels=[np.median(data)], colors='k', linestyles='-')
        #pcm = ax.pcolormesh(target_plot, vmin=np.min(data), vmax=np.max(data), cmap=colormap)
        pcm = ax.pcolormesh(target_plot, norm=colors.LogNorm(vmin=np.min(data), vmax=np.max(data)), cmap=colormap)
        ax.set_yticklabels([]) # no tick numbers in the target plot horizontal axis
        ax.set_xticklabels([]) # no tick numbers in the target plot horizontal axis
        #cbar = fig.colorbar(pcm, ax=ax, extend='both', orientation='vertical')
        #cbar.ax.set_yticklabels(['10', '20', '30', '40', '50' '60'])  # vertically oriented colorbar
        fig.canvas.set_window_title(prefix+"High_Contrast_Depth_Evaluation")
        plt.savefig('%s/%s_frame_%010d.png' % (folder_name, name, idx))
        #plt.show()


def add_to_metrics(idx, metrics, target_, prediction_, mask, event_frame=None, prefix="", rescale=False,
                   debug=False, output_folder=None):
    if len(metrics) == 0:
        metrics = {k: 0 for k in metrics_keywords}

    prediction_mask = (prediction_ >= 0) & (prediction_ <= np.amax(target_[~np.isnan(target_)]))
    depth_mask = (target_ >= 0) & (target_ <= np.amax(target_[~np.isnan(target_)])) # make (target> 3) for mvsec night drives
    #mask = mask & depth_mask & prediction_mask  # no prediction and depth mask needed for simulation data
    eps = 1e-5

    target = target_[mask] #np.where(mask, target_, np.max(target_[~np.isnan(target_)]))# target_[mask] but without lossing shape
    prediction = prediction_[mask] #np.where(mask, prediction_, np.max(target_[~np.isnan(target_)]))# prediction_[mask] but without lossing shape

    if rescale:
        target, prediction = rescale_by_the_median(target, prediction, debug=debug)

    display_high_contrast_colormap(idx, np.where(mask, target_, np.max(target_[~np.isnan(target_)])),
                np.where(mask, prediction_, np.max(target_[~np.isnan(target_)])), prefix=prefix, colormap='tab20c', debug=debug, folder_name=output_folder)

    # thresholds
    ratio = np.max(np.stack([target/(prediction+eps), prediction/(target+eps)]), axis=0)

    new_metrics = {}
    new_metrics[f"{prefix}threshold_delta_1.25"] = np.mean(ratio <= 1.25)
    new_metrics[f"{prefix}threshold_delta_1.25^2"] = np.mean(ratio <= 1.25**2)
    new_metrics[f"{prefix}threshold_delta_1.25^3"] = np.mean(ratio <= 1.25**3)
    
    # abs diff
    log_diff = np.log(target+eps)-np.log(prediction+eps)
    #log_diff = np.abs(log_target - log_prediction)
    abs_diff = np.abs(target-prediction)
    
    new_metrics[f"{prefix}abs_rel_diff"] = abs_rel_diff(prediction, target)  # (abs_diff/(target+eps)).mean()
    new_metrics[f"{prefix}squ_rel_diff"] = squ_rel_diff(prediction, target)  # (abs_diff**2/(target+eps)).mean()
    new_metrics[f"{prefix}RMS_linear"] = rms_linear(prediction, target)  # np.sqrt((abs_diff**2).mean())
    new_metrics[f"{prefix}RMS_log"] = np.sqrt((log_diff**2).mean())
    new_metrics[f"{prefix}SILog"] = scale_invariant_error(np.log(prediction+eps), np.log(target+eps))  # (log_diff**2).mean()-(log_diff.mean())**2
    #new_metrics[f"{prefix}SILog"] = scale_invariant_error(prediction, target)  # (log_diff**2).mean()-(log_diff.mean())**2
    new_metrics[f"{prefix}mean_depth_error"] = mean_error(prediction, target)  # abs_diff.mean()

    new_metrics[f"{prefix}median_diff"] = np.abs(np.median(target) - np.median(prediction))

    for k, v in new_metrics.items():
        metrics[k] += v

    if debug:
        pprint(new_metrics)
        {print ("%s : %f" % (k, v)) for k,v in new_metrics.items()}
        fig, ax = plt.subplots(ncols=3, nrows=4)
        print(target_.shape)
        ax[0, 0].imshow(target_, vmin=0, vmax=200)
        ax[0, 0].set_title("target depth")
        ax[0, 1].imshow(prediction_, vmin=0, vmax=200)
        ax[0, 1].set_title("prediction depth")
        target_debug = target_.copy()
        target_debug[~mask] = 0
        ax[0, 2].imshow(target_debug, vmin=0, vmax=200)
        ax[0, 2].set_title("target depth masked")

        ax[1, 0].imshow(np.log(target_+eps),vmin=0,vmax=np.log(200))
        ax[1, 0].set_title("log target")
        ax[1, 1].imshow(np.log(prediction_+eps),vmin=0,vmax=np.log(200))
        ax[1, 1].set_title("log prediction")
        ax[1, 2].imshow(np.max(np.stack([target_ / (prediction_ + eps), prediction_ / (target_ + eps)]), axis=0))
        ax[1, 2].set_title("max ratio")

        ax[2, 0].imshow(np.abs(np.log(target_ + eps) - np.log(prediction_ + eps)))
        ax[2, 0].set_title("abs log diff")
        ax[2, 1].imshow(np.abs(target_ - prediction_))
        ax[2, 1].set_title("abs diff")
        if event_frame is not None:
            a = np.zeros(event_frame.shape)
            a[:,:,0]= (np.sum(event_frame.astype("float32"), axis=-1)>0)
            a[:,:,1]= np.clip(target_.copy(), 0, 1) 
            ax[2, 2].imshow(a)
            ax[2, 2].set_title("event frame")

        log_diff_ = np.abs(np.log(target_ + eps) - np.log(prediction_ + eps))
        log_diff_[~mask] = 0
        ax[3, 0].imshow(log_diff_)
        ax[3, 0].set_title("abs log diff masked")
        abs_diff_ = np.abs(target_ - prediction_)
        abs_diff_[~mask] = 0
        ax[3, 1].imshow(abs_diff_)
        ax[3, 1].set_title("abs diff masked")
        ax[3, 2].imshow(mask)
        ax[3, 2].set_title("mask frame")

        fig.canvas.set_window_title(prefix+"_Depth_Evaluation")
        plt.show()

    return metrics


if __name__ == "__main__":
    flags = FLAGS()

    reg_factor = flags.reg_factor

    # predicted labels
    prediction_files = sorted(glob.glob(join(flags.predictions_dataset, '*.npy')))
    prediction_files = prediction_files[flags.prediction_offset:]

    target_files = sorted(glob.glob(join(flags.target_dataset, '*.npy')))
    target_files = target_files[flags.target_offset:]

    if flags.event_masks is not "":
        event_frame_files = sorted(glob.glob(join(flags.event_masks, '*png')))
        event_frame_files = event_frame_files[flags.prediction_offset:]

    #prediction_timestamps = np.genfromtxt(join(flags.predictions_dataset, 'data/timestamps.txt'))
    #target_timestamps = np.genfromtxt(join(flags.target_dataset, 'data/timestamps.txt'))

    # Information about the dataset length
    print("len of prediction files", len(prediction_files))
    print("len of target files", len(target_files))
    print(flags.predictions_dataset)
    print(flags.target_dataset)

    if flags.event_masks is not "":
        print("len of events files", len(event_frame_files))

    assert len(prediction_files)>0
    assert len(target_files)>0

    if flags.event_masks is not "":
        use_event_masks = len(event_frame_files)>0
    else:
        use_event_masks = False

    metrics = {}
    metrics2 = []

    num_it = len(prediction_files)

    # the following two lines can be changed to compare the depth with the previous image predictions.
    # for idx in tqdm.tqdm(range(num_it - 1)):
    #    p_file, t_file = prediction_files[idx], target_files[idx+1]
    for idx in tqdm.tqdm(range(num_it)):
        p_file, t_file = prediction_files[idx], target_files[idx]
        # Read absolute scale ground truth
        target_depth = np.load(t_file)

        # Crop depth height according to argument
        target_depth = target_depth[:flags.crop_ymax]

        # Read predicted depth data
        predicted_depth = np.load(p_file)

        # Crop depth height according to argument
        predicted_depth = predicted_depth[:flags.crop_ymax]

        # Convert to the correct scale
        target_depth, predicted_depth = prepare_depth_data(target_depth[0], predicted_depth[0],
                                                           flags.clip_distance, flags.down_scale_factor, reg_factor)

        assert predicted_depth.shape == target_depth.shape

        depth_mask = (np.ones_like(target_depth)>0)
        debug = flags.debug and idx == flags.idx
        metrics = add_to_metrics(idx, metrics, target_depth, predicted_depth, depth_mask, event_frame=None,
                                 prefix="_", rescale=flags.rescale, debug=debug, output_folder=flags.output_folder)

        metrics2.append(eval_metrics(predicted_depth, target_depth))

        for depth_threshold in depth_values:
            depth_threshold_mask = (np.nan_to_num(target_depth) < depth_threshold)
            add_to_metrics(-1, metrics, target_depth, predicted_depth, depth_mask & depth_threshold_mask,
                           prefix=f"_{depth_threshold}_", debug=debug)

        if use_event_masks:
            ev_frame_file = event_frame_files[idx]
            event_frame = cv2.imread(ev_frame_file)
            event_frame = event_frame[:flags.crop_ymax]
            if flags.down_scale_factor < 1.0:
                width = int(event_frame.shape[1] * flags.down_scale_factor)
                height = int(event_frame.shape[0] * flags.down_scale_factor)
                dim = (width, height)
                # resize image
                event_frame = cv2.resize(event_frame, dim, interpolation=cv2.INTER_LINEAR)

            event_mask = (np.sum(event_frame.astype("float32"), axis=-1)>0)
            assert event_mask.shape == target_depth.shape
            add_to_metrics(-1, metrics, target_depth, predicted_depth, event_mask, event_frame = event_frame,
                           prefix="event_masked_", rescale=flags.rescale, debug=debug)

            for depth_threshold in depth_values:
                depth_threshold_mask = np.nan_to_num(target_depth) < depth_threshold
                #$debug=True
                add_to_metrics(-1, metrics, target_depth, predicted_depth, event_mask & depth_threshold_mask, event_frame = event_frame, prefix=f"event_masked_{depth_threshold}_", rescale=flags.rescale, debug=debug)


    {print("%s : %f" % (k, v/num_it)) for k,v in metrics.items()}
    print("----------------------------------------------")
    {print ("%f" % (v/num_it)) for _,v in metrics.items()}

    print("total metrics: ", np.sum(np.array(metrics2), 0) / len(metrics2))
