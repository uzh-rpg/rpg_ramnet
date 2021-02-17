import os
import json
import logging
import argparse
import torch
from model.model import *
from model.loss import *
from model.metric import *
from torch.utils.data import DataLoader, ConcatDataset
from data_loader.dataset import *
from trainer.lstm_trainer import LSTMTrainer
from utils.data_augmentation import Compose, RandomRotationFlip, RandomCrop, CenterCrop
from os.path import join
import cv2
import matplotlib as mpl
import matplotlib.cm as cm
from train import concatenate_subfolders
import matplotlib.pyplot as plt
import numpy as np
import time

logging.basicConfig(level=logging.INFO, format='')

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def eval_metrics(output, target):
    metrics = [mse, abs_rel_diff, scale_invariant_error, median_error, mean_error, rms_linear]
    acc_metrics = np.zeros(len(metrics))
    output = output.cpu().data.numpy()
    target = target.cpu().data.numpy()
    for i, metric in enumerate(metrics):
        acc_metrics[i] += metric(output, target)
    return acc_metrics

def make_colormap(img, color_mapper):
    #color_map = np.nan_to_num(img[0])
    #print("max min color map: ", np.max(color_map), np.min(color_map))
    img = np.nan_to_num(img, nan=1)
    color_map_inv = np.ones_like(img[0]) * np.amax(img[0]) - img[0]
    color_map_inv = color_map_inv / np.amax(color_map_inv)
    color_map_inv = np.nan_to_num(color_map_inv)
    color_map_inv = color_mapper.to_rgba(color_map_inv)
    color_map_inv[:, :, 0:3] = color_map_inv[:, :, 0:3][..., ::-1]
    return color_map_inv

def main(config, initial_checkpoint, output_folder, data_folder):
    train_logger = None

    # L = config['trainer']['sequence_length']
    # assert(L > 0)
    L = 1
    calculate_scale = False

    dataset_type, base_folder, event_folder, depth_folder, frame_folder = {}, {}, {}, {}, {}
    proba_pause_when_running, proba_pause_when_paused = {}, {}
    step_size = {}
    clip_distance = {}
    scale_factor = {}
    every_x_rgb_frame = {}
    baseline = {}
    nbr_of_bins = {}
    nbr_of_events_per_voxelgrid = {}
    event_loader = {}
    total_metrics = []

    # this will raise an exception is the env variable is not set
    preprocessed_datasets_folder = os.environ['PREPROCESSED_DATASETS_FOLDER']

    if output_folder:
        ensure_dir(output_folder)
        depth_dir = join(output_folder, "depth")
        npy_dir = join(output_folder, "npy")
        color_map_dir = join(output_folder, "color_map")
        inputs_dir = join(output_folder, "inputs")
        gt_dir_grey = join(output_folder, "ground_truth/grey")
        gt_dir_color_map = join(output_folder, "ground_truth/color_map")
        gt_dir_npy = join(output_folder, "ground_truth/npy")
        semantic_seg_dir_npy = join(output_folder, "semantic_seg/npy")
        semantic_seg_dir_frames = join(output_folder, "semantic_seg/frames")
        video_pred = join(output_folder, "video/predictions")
        video_gt = join(output_folder, "video/gt")
        video_inputs = join(output_folder, "video/inputs")
        ensure_dir(depth_dir)
        ensure_dir(npy_dir)
        ensure_dir(color_map_dir)
        ensure_dir(inputs_dir)
        ensure_dir(gt_dir_grey)
        ensure_dir(gt_dir_color_map)
        ensure_dir(gt_dir_npy)
        ensure_dir(semantic_seg_dir_npy)
        ensure_dir(semantic_seg_dir_frames)
        ensure_dir(video_pred)
        ensure_dir(video_gt)
        ensure_dir(video_inputs)
        print('Will write images to: {}'.format(depth_dir))
        # timestamps_file = open(join(depth_dir, 'timestamps.txt'), 'a')

    use_phased_arch = config['use_phased_arch']

    dataset_type['validation'] = config['data_loader']['validation']['type']
    # change path here if test.py needs to be run on validation or training data
    if data_folder is not None:
        base_folder['validation'] = join(preprocessed_datasets_folder, data_folder)
    else:
        base_folder['validation'] = join(preprocessed_datasets_folder, 'mvsec/mvsec_daniel/validation/')
    #base_folder['validation'] = join(preprocessed_datasets_folder, 'mvsec/mvsec_daniel/test_outdoor_night1/')
    #base_folder['validation'] = join(preprocessed_datasets_folder, 'mvsec')
    event_folder['validation'] = config['data_loader']['validation']['event_folder']
    depth_folder['validation'] = config['data_loader']['validation']['depth_folder']
    frame_folder['validation'] = config['data_loader']['validation']['frame_folder']
    proba_pause_when_running['validation'] = config['data_loader']['validation']['proba_pause_when_running']
    proba_pause_when_paused['validation'] = config['data_loader']['validation']['proba_pause_when_paused']
    scale_factor['validation'] = config['data_loader']['validation']['scale_factor']

    try:
        #step_size['validation'] = config['data_loader']['validation']['step_size']
        step_size['validation'] = 1
    except KeyError:
        step_size['validation'] = 1

    try:
        clip_distance['validation'] = config['data_loader']['validation']['clip_distance']
    except KeyError:
        clip_distance['validation'] = 100.0
        print("Clip distance not loaded properly!")

    try:
        every_x_rgb_frame['validation'] = config['data_loader']['validation']['every_x_rgb_frame']
    except KeyError:
        every_x_rgb_frame['validation'] = 1
        print("Every_x_rgb_frame not loaded properly!")

    try:
        baseline['validation'] = config['data_loader']['validation']['baseline']
    except KeyError:
        baseline['validation'] = False
        print("Baseline not loaded properly!")

    try:
        event_loader['validation'] = config['data_loader']['validation']['event_loader']
    except KeyError:
        event_loader['validation'] = "voxelgrids"

    try:
        nbr_of_events_per_voxelgrid['validation'] = config['data_loader']['validation']['nbr_of_events_per_voxelgrid']
    except KeyError:
        nbr_of_events_per_voxelgrid['validation'] = 0

    try:
        nbr_of_bins['validation'] = config['data_loader']['validation']['nbr_of_bins']
    except KeyError:
        nbr_of_bins['validation'] = 5

    normalize = config['data_loader'].get('normalize', True)
    loss_composition = config['trainer']['loss_composition']

    test_dataset = concatenate_subfolders(base_folder['validation'],
                                          None,
                                          dataset_type['validation'],
                                          None,
                                          event_folder['validation'],
                                          depth_folder['validation'],
                                          frame_folder['validation'],
                                          sequence_length=L,
                                          # change transform to test on different resolution
                                          #transform=CenterCrop(224),
                                          transform=CenterCrop([256, 344]),
                                          proba_pause_when_running=proba_pause_when_running['validation'],
                                          proba_pause_when_paused=proba_pause_when_paused['validation'],
                                          step_size=step_size['validation'],
                                          clip_distance=clip_distance['validation'],
                                          every_x_rgb_frame=every_x_rgb_frame['validation'],
                                          normalize=normalize,
                                          scale_factor=scale_factor['validation'],
                                          use_phased_arch=use_phased_arch,
                                          baseline=baseline['validation'],
                                          loss_composition=loss_composition,
                                          event_loader=event_loader['validation'],
                                          nbr_of_events_per_voxelgrid=nbr_of_events_per_voxelgrid['validation'],
                                          nbr_of_bins=nbr_of_bins['validation'],
                                          dataset_idx_flag=True
                                          )

    config['model']['gpu'] = config['gpu']
    config['model']['every_x_rgb_frame'] = config['data_loader']['train']['every_x_rgb_frame']
    config['model']['baseline'] = config['data_loader']['train']['baseline']
    config['model']['loss_composition'] = config['trainer']['loss_composition']

    model = eval(config['arch'])(config['model'])

    if initial_checkpoint is not None:
        print('Loading initial model weights from: {}'.format(initial_checkpoint))
        checkpoint = torch.load(initial_checkpoint)
        if use_phased_arch:
            C, (H, W) = config["model"]["num_bins_events"], config["model"]["spatial_resolution"]
            dummy_input = torch.Tensor(1, C, H, W)
            times = torch.Tensor(1)
            _ = model.forward(dummy_input, times=times, prev_states=None)
        model.load_state_dict(checkpoint['state_dict'])

    model.summary()

    gpu = torch.device('cuda:' + str(config['gpu']))
    model.to(gpu)

    reg_factor = config['data_loader']['train']['reg_factor']

    model.eval()

    N = len(test_dataset)

    video_idx = 0

    if calculate_scale:
        scale = np.empty(N)

    # construct color mapper
    # get groundtruth that is not at the beginning
    item, dataset_idx = test_dataset[20]
    img = item[0][-2]['depth'].cpu().numpy()
    img = np.nan_to_num(img, nan=1.0)
    color_map_inv = np.ones_like(img[0]) * np.amax(img[0]) - img[0]
    color_map_inv = np.nan_to_num(color_map_inv, nan=1.0)
    color_map_inv = color_map_inv / np.amax(color_map_inv)
    color_map_inv = np.nan_to_num(color_map_inv)
    vmax = np.percentile(color_map_inv, 95)
    # print("color_mapper: ", color_map_inv.min(), vmax, color_map_inv.max())
    normalizer = mpl.colors.Normalize(vmin=color_map_inv.min(), vmax=vmax)
    color_mapper_overall = cm.ScalarMappable(norm=normalizer, cmap='magma')
    time_measurements = []

    with torch.no_grad():
        idx = 0
        prev_dataset_idx = -1
        # for batch_idx, sequence in enumerate(data_loader):

        while idx < N:
            item, dataset_idx = test_dataset[idx]

            '''#if idx in [0,1,2,40,100,500]:
            fig, ax = plt.subplots(ncols=3, nrows=1)
            ax[0].imshow(item[0]['image'][0])
            ax[0].set_title("image")
            ax[1].imshow((torch.sum(item[0]['events4'], axis=0).numpy()))
            ax[1].set_title("events4")
            ax[2].imshow((10*item[0]['image'][0] - torch.sum(item[0]['events4'], axis=0)).numpy())
            ax[2].set_title("events 4 - image")

            plt.show()'''

            if dataset_idx > prev_dataset_idx:
                # reset internal states for new sequence
                #print("reset states")
                prev_states_lstm = [{'image_last': None, 'events_last': None}]
                prev_super_states = [None]
                sequence_idx = 0

            #print(idx, sequence_idx, dataset_idx)

            # new_events, new_image, new_target, times = to_input_and_target(item[0], gpu, use_phased_arch)
            # the output of the network is a [N x 1 x H x W] tensor containing the image prediction
            #t0 = time.time()
            new_predicted_targets, new_super_states, new_states_lstm = model(item,
                                                                             prev_super_states,
                                                                             prev_states_lstm)
            '''t1 = time.time() - t0
            time_measurements.append(t1)
            print("Time elapsed: ", t1)  # CPU seconds elapsed (floating point)'''

            if idx > 0 and idx < 2:
                #if idx == 0:
                index_0 = 0
                fig, ax = plt.subplots(ncols=every_x_rgb_frame['validation']+1, nrows=4)
                for i, key in enumerate(new_predicted_targets[0].keys()):
                    if "last" not in key:
                        ax[0, index_0].imshow(new_predicted_targets[0][key][0].cpu().numpy()[0])
                        ax[0, index_0].set_title("prediction " + key)
                        index_0 += 1
                index_1 = 0
                index_2 = 0
                for i, value in enumerate(item[0]):
                    key = list(value.keys())[0]
                    print("preview: ", key)
                    if "depth" in key:
                        ax[1, index_1].imshow(value[key][0].cpu().numpy())
                        ax[1, index_1].set_title("groundtruth " + key)
                        index_1 += 1
                    elif key != "img_loss":  # and "last" not in key:
                        ax[2, index_2].imshow(torch.sum(value[key], dim=0).cpu().numpy())
                        ax[2, index_2].set_title("input " + key)
                        index_2 += 1
                plt.show()

            if output_folder and sequence_idx > 1: #and "image_last" in new_predicted_targets[0].keys():
                # don't save the first 2 predictions such that the temporal dependencies of the network are settled.
                # only save prediction if datapackage contains an image. like this, the rgb baseline and statenet
                # are evaluated on the same amount of datapoints.
                for key, img in new_predicted_targets[0].items():

                    # print("metrics of index ", idx, ": ", metrics)
                    img = img[0].cpu().numpy()

                    # save depth image
                    depth_dir_key = join(depth_dir, key)
                    ensure_dir(depth_dir_key)
                    cv2.imwrite(join(depth_dir_key, 'frame_{:010d}.png'.format(idx)), img[0][:, :, None] * 255.0)

                    # save numpy
                    npy_dir_key = join(npy_dir, key)
                    ensure_dir(npy_dir_key)
                    data = img
                    np.save(join(npy_dir_key, 'depth_{:010d}.npy'.format(idx)), data)

                    # save color map
                    color_map_dir_key = join(color_map_dir, key)
                    ensure_dir(color_map_dir_key)
                    color_map = make_colormap(img, color_mapper_overall)
                    cv2.imwrite(join(color_map_dir_key, 'frame_{:010d}.png'.format(idx)), color_map * 255.0)

                for i, value in enumerate(item[0]):
                    key = list(value.keys())[0]
                    if 'depth' in key:
                        # save GT images grey
                        gt_dir_grey_key = join(gt_dir_grey, key)
                        ensure_dir(gt_dir_grey_key)
                        img = value[key].cpu().numpy()
                        cv2.imwrite(join(gt_dir_grey_key, 'frame_{:010d}.png'.format(idx)), img[0][:, :, None] * 255.0)

                        # save GT images color map
                        gt_dir_cm_key = join(gt_dir_color_map, key)
                        ensure_dir(gt_dir_cm_key)
                        color_map = make_colormap(img, color_mapper_overall)
                        cv2.imwrite(join(gt_dir_cm_key, 'frame_{:010d}.png'.format(idx)), color_map * 255.0)

                        # save GT to numpy array
                        gt_dir_npy_key = join(gt_dir_npy, key)
                        ensure_dir(gt_dir_npy_key)
                        np.save(join(gt_dir_npy_key, 'frame_{:010d}.npy'.format(idx)), img)
                    elif 'semantic' in key:
                        # save semantic seg numpy array
                        img = value[key].cpu().numpy()[0]
                        semantic_seg_dir_npy_key = join(semantic_seg_dir_npy, key)
                        ensure_dir(semantic_seg_dir_npy_key)
                        np.save(join(semantic_seg_dir_npy_key, 'frame_{:010d}.npy'.format(idx)), img)
                        # save semantic seg frame
                        semantic_seg_dir_frames_key = join(semantic_seg_dir_frames, key)
                        ensure_dir(semantic_seg_dir_frames_key)
                        cv2.imwrite(join(semantic_seg_dir_frames_key, 'frame_{:010d}.png'.format(idx)), img)
                    elif 'events' in key or 'image' in key:
                        # save inputs
                        img = value[key].cpu().numpy()
                        input_dir_key = join(inputs_dir, key)
                        ensure_dir(input_dir_key)
                        input_data = np.sum(img, axis=0)
                        if "event" in key:
                            # input_data = np.ones_like(input_data) * 0.5
                            # negativ_input = np.where(input_data <= 0, np.ones_like(input_data), np.zeros_like(input_data))
                            negativ_input = np.where(input_data <= -0.5, 1.0, 0.0)
                            positiv_input = np.where(input_data > 0.9, 1.0, 0.0)
                            zeros_input = np.zeros_like(input_data)
                            total_image = np.concatenate(
                                (negativ_input[:, :, None], zeros_input[:, :, None], positiv_input[:, :, None]), axis=2)
                            '''fig, ax = plt.subplots(ncols=1, nrows=4)
                            ax[0].imshow(negativ_input)
                            ax[0].set_title("negativ input")
                            ax[1].imshow(positiv_input)
                            ax[1].set_title("positive input")
                            ax[2].imshow(zeros_input)
                            ax[2].set_title("zeros input")
                            ax[3].imshow(total_image)
                            ax[3].set_title("total input")
                            plt.show()'''
                            cv2.imwrite(join(input_dir_key, 'frame_{:010d}.png'.format(idx)),
                                        total_image * 255.0)

                            #cv2.imwrite(join(video_inputs, 'frame_{:010d}.png'.format(video_idx)), total_image * 255.0)
                        else:
                            cv2.imwrite(join(input_dir_key, 'frame_{:010d}.png'.format(idx)),
                                        input_data[:, :, None] * 255.0)

                # save data for video of consecutive inputs

                for i, value in enumerate(item[0]):
                    key = list(value.keys())[0]
                    if "depth" not in key and key != "img_loss":
                        prediction = new_predicted_targets[0][key][0].cpu().numpy()
                        gt_data = item[0][-2]["depth"].cpu().numpy()
                        input_data = value[key].cpu().numpy()
                        # save data
                        cm_prediction = make_colormap(prediction, color_mapper_overall)
                        cv2.imwrite(join(video_pred, 'frame_{:010d}.png'.format(video_idx)), cm_prediction * 255.0)
                        cm_gt_data = make_colormap(gt_data, color_mapper_overall)
                        cv2.imwrite(join(video_gt, 'frame_{:010d}.png'.format(video_idx)), cm_gt_data * 255.0)
                        input_data = np.sum(input_data, axis=0)
                        cv2.imwrite(join(video_inputs, 'frame_{:010d}.png'.format(video_idx)), input_data[:, :, None] * 255.0)

                        video_idx += 1

                if idx % 100 == 0:
                    print("saved image ", idx)

            if calculate_scale:
                for key, img in new_predicted_targets[0].items():
                    key_target = "depth"

                    prediction = img[0][0].cpu().numpy()
                    target = item[0][-2][key_target][0].cpu().numpy()
                    # change to metric space
                    prediction = np.exp(reg_factor * (prediction - np.ones((prediction.shape[0], prediction.shape[1]),
                                                                           dtype=np.float32)))
                    target = np.exp(reg_factor * (target - np.ones((target.shape[0], target.shape[1]),
                                                                   dtype=np.float32)))
                    target *= clip_distance['validation']
                    prediction *= clip_distance['validation']
                    #mask = np.ones_like(target) > 0
                    mask = np.isnan(target)
                    mask = ~mask

                    scale[idx] = np.sum(prediction[mask] * target[mask]) / np.sum(prediction[mask] * prediction[mask])
                if idx % 100 == 0:
                    print("current scale", np.mean(scale))

            prev_super_states = new_super_states
            prev_states_lstm = new_states_lstm
            sequence_idx += 1
            prev_dataset_idx = dataset_idx
            idx += 1

        if calculate_scale:
            total_scale = np.mean(scale)
            # print(scale)
            print("total scale: ", total_scale)
            print("min scale: ", np.min(scale))
            print("max scale: ", np.max(scale))

        #print("Time statistics: ", np.mean(np.asarray(time_measurements)), min(time_measurements),
        #      max(time_measurements))

        # total metrics:
        # print("total metrics: ", np.sum(np.array(total_metrics), 0) / len(total_metrics))


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(
        description='Learning DVS Image Reconstruction')
    parser.add_argument('--path_to_model', type=str,
                        help='path to the model weights',
                        default='')
    parser.add_argument('--output_path', type=str,
                        help='path to folder for saving outputs',
                        default='')
    parser.add_argument('--config', type=str,
                        help='path to config. If not specified, config from model folder is taken',
                        default=None)
    parser.add_argument('--data_folder', type=str,
                        help='path to folder of data to be tested',
                        default=None)

    args = parser.parse_args()

    if args.config is None:
        head_tail = os.path.split(args.path_to_model)
        config = json.load(open(os.path.join(head_tail[0], 'config.json')))
    else:
        config = json.load(open(args.config))

    main(config, args.path_to_model, args.output_path, args.data_folder)
