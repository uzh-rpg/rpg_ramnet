# -*- coding: utf-8 -*-
"""
Dataset classes
"""

from torch.utils.data import Dataset
from .event_dataset import VoxelGridDataset, FrameDataset
from .dataset_asynchronous import SynchronizedFramesEventsRawDataset
from skimage import io
from os.path import join
import numpy as np
from utils.util import first_element_greater_than, last_element_less_than
import random
import glob
import torch
import torch.nn as nn
import torch.nn.functional as f
from math import fabs
import cv2
import matplotlib.pyplot as plt


class SequenceSynchronizedFramesEventsDataset(Dataset):
    """Load sequences of time-synchronized {event tensors + depth} from a folder."""

    def __init__(self, base_folder, event_folder, depth_folder='frames', frame_folder='rgb', flow_folder='flow', semantic_folder='semantic/data/',
                 start_time=0.0, stop_time=0.0,
                 sequence_length=2, transform=None,
                 proba_pause_when_running=0.0, proba_pause_when_paused=0.0,
                 step_size=20,
                 clip_distance=100.0,
                 normalize=True,
                 scale_factor=1.0,
                 use_phased_arch=False,
                 every_x_rgb_frame=1,
                 baseline=False,
                 loss_composition=False,
                 reg_factor=5.7,
                 recurrency=True):
        assert(sequence_length > 0)
        assert(step_size > 0)
        assert(clip_distance > 0)
        self.L = sequence_length
        if not recurrency:
            self.dataset = SynchronizedFramesEventsRawDataset(base_folder, event_folder, depth_folder, frame_folder,
                                                           flow_folder, semantic_folder, start_time, stop_time,
                                                           clip_distance, every_x_rgb_frame, transform,
                                                           normalize=normalize, use_phased_arch=use_phased_arch,
                                                           baseline=baseline, loss_composition=loss_composition)
        else:
            self.dataset = SynchronizedFramesEventsDataset(base_folder, event_folder, depth_folder, frame_folder,
                                                           flow_folder, semantic_folder, start_time, stop_time,
                                                           clip_distance, every_x_rgb_frame, transform,
                                                           normalize=normalize, use_phased_arch=use_phased_arch,
                                                           baseline=baseline, loss_composition=loss_composition,
                                                           reg_factor=reg_factor, recurrency=recurrency)
        self.event_dataset = self.dataset.event_dataset
        self.step_size = step_size
        self.every_x_rgb_frame = every_x_rgb_frame
        if self.L * self.every_x_rgb_frame >= self.dataset.length:
            self.length = 0
        else:
            self.length = (self.dataset.length - self.L * self.every_x_rgb_frame) // self.step_size\
                          // self.every_x_rgb_frame + 1

        self.proba_pause_when_running = proba_pause_when_running
        self.proba_pause_when_paused = proba_pause_when_paused
        self.scale_factor = scale_factor
        self.use_phased_arch = use_phased_arch
        self.baseline = baseline

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        """ Returns a list containing synchronized events <-> frame pairs
            [e_{i-L} <-> I_{i-L},
             e_{i-L+1} <-> I_{i-L+1},
            ...,
            e_{i-1} <-> I_{i-1},
            e_i <-> I_i]
        """
        assert(i >= 0)
        assert(i < self.length)

        # generate a random seed here, that we will pass to the transform function
        # of each item, to make sure all the items in the sequence are transformed
        # in the same way
        seed = random.randint(0, 2**32)

        # data augmentation: add random, virtual "pauses",
        # i.e. zero out random event tensors and repeat the last frame
        sequence = []

        # add the first element (i.e. do not start with a pause)
        k = 0
        j = i * self.step_size
        item = self.dataset.__getitem__(j, seed)
        sequence.append(item)

        paused = False
        for n in range(self.L - 1):

            # decide whether we should make a "pause" at this step
            # the probability of "pause" is conditioned on the previous state (to encourage long sequences)
            u = np.random.rand()
            if paused:
                probability_pause = self.proba_pause_when_paused
            else:
                probability_pause = self.proba_pause_when_running
            paused = (u < probability_pause)

            if paused:
                # add a tensor filled with zeros, paired with the last frame
                # do not increase the counter
                item = self.dataset.__getitem__(j + k, seed)
                item['events'].fill_(0.0)
                if 'flow' in item:
                    item['flow'].fill_(0.0)
                sequence.append(item)
            else:
                # normal case: append the next item to the list
                k += 1
                item = self.dataset.__getitem__(j + k, seed)
                sequence.append(item)

        # down sample data
        if self.scale_factor < 1.0:
            for data_items in sequence:
                for k, item in data_items.items():
                    if k is not "times" and k is not "batchlength_events":
                        item = item[None]
                        if "semantic" in k:
                            item = f.interpolate(item, scale_factor=self.scale_factor,
                                                 recompute_scale_factor=False)
                        else:
                            item = f.interpolate(item, scale_factor=self.scale_factor, mode='bilinear',
                                                 recompute_scale_factor=False, align_corners=False)
                        item = item[0]
                        data_items[k] = item
        return sequence


class SynchronizedFramesEventsDataset(Dataset):
    """Loads time-synchronized event tensors and depth from a folder.

    This Dataset class iterates through all the event tensors and returns, for each tensor,
    a dictionary of the form:

        {'depth': frame, 'events': events, 'flow': disp_01, 'semantic': semantic}

    where:

    * depth is a H x W tensor containing the first frame whose timestamp >= event tensor
    * events is a C x H x W tensor containing the event data
    * flow is a 2 x H x W tensor containing the flow (displacement) from the current frame to the last frame
    * semantic is a 1 x H x W tensor containing the semantic labels 

    This loader assumes that each event tensor can be uniquely associated with a frame.
    For each event tensor with timestamp e_t, the corresponding frame is the first frame whose timestamp f_t >= e_t

    """

    def __init__(self, base_folder, event_folder, depth_folder='frames', frame_folder='rgb', flow_folder='flow', semantic_folder='semantic',
                 start_time=0.0, stop_time=0.0, clip_distance=100.0, every_x_rgb_frame=1,
                 transform=None,
                 normalize=True,
                 use_phased_arch=False,
                 baseline=False,
                 loss_composition=False,
                 reg_factor=5.7,
                 recurrency=True):
        '''print((base_folder, event_folder, depth_folder, frame_folder, flow_folder, semantic_folder, \
                 start_time, stop_time, clip_distance, every_x_rgb_frame, \
                 transform, normalize, use_phased_arch, baseline))'''

        self.base_folder = base_folder
        self.depth_folder = join(self.base_folder, depth_folder if depth_folder is not None else 'frames')
        self.frame_folder = join(self.base_folder, frame_folder if frame_folder is not None else 'rgb')
        self.flow_folder = join(self.base_folder, flow_folder if flow_folder is not None else 'flow')
        self.semantic_folder = join(self.base_folder, semantic_folder if semantic_folder is not None else 'semantic')
        self.transform = transform
        self.event_dataset = VoxelGridDataset(base_folder, event_folder,
                                                  start_time, stop_time,
                                                  transform=self.transform,
                                                  normalize=normalize)

        self.eps = 1e-06
        self.clip_distance = clip_distance
        self.use_phased_arch = use_phased_arch
        self.every_x_rgb_frame = every_x_rgb_frame
        self.baseline = baseline
        self.loss_composition = loss_composition
        self.reg_factor = reg_factor
        self.recurrency = recurrency

        self.test = False

        if "mvsec" in base_folder or "javi" in base_folder:
            self.use_mvsec = True
        else:
            self.use_mvsec = False

        # Load the stamp files
        self.stamps = np.loadtxt(
            join(self.depth_folder, 'timestamps.txt'))[:, 1]

        if self.use_mvsec and not "javi" in self.base_folder:
            self.stamps = self.stamps[1:]

        # shift the frame timestamps by the same amount as the event timestamps
        self.stamps -= self.event_dataset.initial_stamp

        # length = total number of datapoints, not equal to length of final dataset due to every_x_rgb_frame & step size
        self.length = len(self.event_dataset)

        # Check that the frame timestamps are unique and sorted
        assert(np.alltrue(np.diff(self.stamps) > 0)
               ), "frame timestamps are not unique and monotonically increasing"

        # Check that the latest frame in the dataset has a timestamp >= the latest event frame
        '''assert(
            self.stamps[-1] >= self.event_dataset.get_last_stamp())'''

    def __len__(self):
        return self.length

    def __getitem__(self, i, seed=None):
        #def __getitem__(self, i, seed=None, reg_factor=5.70378): 
        reg_factor = self.reg_factor
        assert(i >= 0)
        assert(i < (self.length // self.every_x_rgb_frame))
        item = {}

        def rgb2gray(rgb):
            return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.float32)

        def nan_helper(self, y):
            """Helper to handle indices and logical indices of NaNs.

            Input: 
                - y, 1d numpy array with possible NaNs
            Output:
                - nans, logical indices of NaNs
                - index, a function, with signature indices= index(logical_indices),
                to convert logical indices of NaNs to 'equivalent' indices
            Example:
                >>> # linear interpolation of NaNs
                >>> nans, x= nan_helper(y)
                >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
            """
            return np.isnan(y), lambda z: z.nonzero()[0]

        for k in range(0, self.every_x_rgb_frame):
            j = i * self.every_x_rgb_frame + k
            event_timestamp = self.event_dataset.get_stamp_at(j)

            # Find the index of the first frame whose timestamp is >= event timestamp
            (frame_idx, frame_timestamp) = first_element_greater_than(self.stamps, event_timestamp)
            assert(frame_idx >= 0)
            assert(frame_idx < len(self.stamps))
            # assert(frame_timestamp >= event_timestamp)
            assert(frame_timestamp - event_timestamp < 0.00001)
            # tol = 0.01
            # if fabs(frame_timestamp - event_timestamp) > tol:
            #     print(
            #         'Warning: frame_timestamp and event_timestamp differ by more than tol ({} s)'.format(tol))
            #     print('frame_timestamp = {}, event_timestamp = {}'.format(
            #         frame_timestamp, event_timestamp))

            #frame = io.imread(join(self.depth_folder, 'frame_{:010d}.png'.format(frame_idx)),
            #                  as_gray=False).astype(np.float32) / 255.

            if seed is None:
                # if no specific random seed was passed, generate our own.
                # otherwise, use the seed that was passed to us
                seed = random.randint(0, 2**32)

            # Get the event tensor from the event dataset loader
            # Note that we pass the transform seed to ensure the same transform is
            if self.baseline != 'rgb':
                events = self.event_dataset.__getitem__(j, seed)

            # Load numpy depth ground truth frame
            if self.use_mvsec:
                frame = np.load(join(self.depth_folder, 'depth_{:010d}.npy'.format(frame_idx))).astype(np.float32)
            else:
                path_depthframe = glob.glob(self.depth_folder + '/*_{:04d}_depth.npy'.format(frame_idx))
                frame = np.load(path_depthframe[0]).astype(np.float32)

            # if np.isnan(frame).sum()>0:
                # events_mask = (torch.sum(events["events"], dim=0).unsqueeze(0))>0
                # frame[~events_mask] = 0
                # nans, x= nan_helper(frame)
                # frame[nans]= np.interp(x(nans), x(~nans), frame[~nans])
            # Clip to maximum distance
            frame = np.clip(frame, 0.0, self.clip_distance)
            # Normalize
            frame = frame / self.clip_distance
            #frame = frame / np.amax(frame[~np.isnan(frame)])
            # div = abs(np.min(np.log(frame+self.eps)))
            # Convert to log depth
            frame = 1.0 + np.log(frame) / reg_factor
            # Clip between 0 and 1.0
            frame = frame.clip(0, 1.0)

            if len(frame.shape) == 2:  # [H x W] grayscale image -> [H x W x 1]
                frame = np.expand_dims(frame, -1)

            frame = np.moveaxis(frame, -1, 0)  # H x W x C -> C x H x W
            frame = torch.from_numpy(frame)  # numpy to tensor

            if self.transform:
                random.seed(seed)
                frame = self.transform(frame)


            # if test script is run, semantic segmentation is saved for later evaluation
            if self.test:
                segmask_path = glob.glob(self.semantic_folder + '/*_{:04d}_gt_labelIds.png'.format(frame_idx))
                seg_mask = cv2.imread(segmask_path[0])[:, :, 0].astype(np.float32)
                seg_mask = torch.tensor(seg_mask).unsqueeze(0)  # [1 x H x W]
                if self.transform:
                    random.seed(seed)
                    seg_mask = self.transform(seg_mask)

            if self.use_phased_arch:
                timestamp = torch.from_numpy(np.asarray([event_timestamp]).astype(np.float32))
                # timestamp = torch.tensor([event_timestamp])

            if not bool(self.baseline) or \
                    (self.baseline == 'e' and self.loss_composition == "image" and k < self.every_x_rgb_frame - 1):
                # for baseline e with sparse supervision, last event tensor of data package will be saved
                # in the "image" entry.
                item['events{}'.format(k)] = events['events']
                item['depth_events{}'.format(k)] = frame
                if self.test:
                    item['semantic_seg_{}'.format(k)] = seg_mask
                if self.use_phased_arch:
                    item['times_events{}'.format(k)] = timestamp

            if self.baseline == "ergb0" and k < self.every_x_rgb_frame - 1:
                if k == 0:  # remove this if statement for ergb ideal!
                    if frame_idx < self.every_x_rgb_frame:
                        # use black image for beginning of dataset
                        last_gray_frame = torch.zeros_like(frame)
                    else:
                        if self.use_mvsec:
                            rgb_frame = io.imread(
                                join(self.frame_folder, 'frame_{:010d}.png'.format(frame_idx - (k + 1))),
                                as_gray=False).astype(np.float32)
                        else:
                            path_rgbframe = glob.glob(self.frame_folder + '/*_{:04d}_image.png'.format(frame_idx-(k+1)))
                            #path_rgbframe = glob.glob(self.frame_folder + '/*_{:04d}_image.png'.format(frame_idx))  # ergb0 ideal
                            rgb_frame = io.imread(path_rgbframe[0], as_gray=False).astype(np.float32)

                        if rgb_frame.shape[2] > 1:
                            last_gray_frame = rgb2gray(rgb_frame)  # [H x W]

                        last_gray_frame /= 255.0  # normalize
                        last_gray_frame = np.expand_dims(last_gray_frame, axis=0)  # expand to [1 x H x W]
                        last_gray_frame = torch.from_numpy(last_gray_frame)
                        if self.transform:
                            random.seed(seed)
                            last_gray_frame = self.transform(last_gray_frame)
                '''fig, ax = plt.subplots(ncols=1, nrows=1)
                #ax.imshow(torch.sum(torch.cat((events["events"], last_gray_frame), axis=0), axis=0))
                ax.imshow(last_gray_frame[0])
                ax.set_title("dataitem for frameindex {}".format(frame_idx))
                #plt.show()'''

                item['events{}'.format(k)] = torch.cat((events["events"], last_gray_frame), axis=0)
                item['depth_events{}'.format(k)] = frame
                if self.use_phased_arch:
                    item['times_events{}'.format(k)] = timestamp
            if k == self.every_x_rgb_frame - 1:
                # Get RGB frame
                if self.frame_folder is not None:
                    try:
                        if self.use_mvsec:
                            rgb_frame = io.imread(join(self.frame_folder, 'frame_{:010d}.png'.format(frame_idx)),
                                                                          as_gray=False).astype(np.float32)
                        else:
                            path_rgbframe = glob.glob(self.frame_folder + '/*_{:04d}_image.png'.format(frame_idx))
                            rgb_frame = io.imread(path_rgbframe[0], as_gray=False).astype(np.float32)

                        if len(rgb_frame.shape) > 2:
                            if rgb_frame.shape[2] > 1:
                                gray_frame = rgb2gray(rgb_frame)  # [H x W]
                        else:
                            gray_frame = rgb_frame

                        gray_frame /= 255.0  # normalize
                        gray_frame = np.expand_dims(gray_frame, axis=0)  # expand to [1 x H x W]
                        gray_frame = torch.from_numpy(gray_frame)
                        if self.transform:
                            random.seed(seed)
                            gray_frame = self.transform(gray_frame)

                        # Combine events with grayscale frames
                        # events["events"] = torch.cat((events["events"], gray_frame), axis=0)
                    except FileNotFoundError:
                        gray_frame = None

                if not bool(self.baseline) or self.baseline == 'rgb':
                    item['image'] = gray_frame
                elif self.baseline == 'ergb' or self.baseline == 'ergb0':
                    # for testing of ergb baseline, e=0 should be equal to rgb baseline
                    # item['image'] = torch.cat((torch.zeros_like(events["events"]), gray_frame), axis=0)
                    item['image'] = torch.cat((events["events"], gray_frame), axis=0)
                elif self.baseline == 'e':
                    item['image'] = events['events']
                item['depth_image'] = frame
                if self.use_phased_arch:
                    item['times_image'] = timestamp
        return item