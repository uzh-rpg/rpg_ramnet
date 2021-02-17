# -*- coding: utf-8 -*-
"""
Dataset classes
"""

from torch.utils.data import Dataset
from .event_dataset import VoxelGridDataset, FrameDataset, RawEventsDataset
from skimage import io
from os.path import join
import numpy as np
from utils.util import first_element_greater_than, last_element_less_than
import random
import glob
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data.dataloader import default_collate
from math import fabs


class SynchronizedFramesEventsRawDataset(Dataset):
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
                 nbr_of_events_per_voxelgrid=0,
                 nbr_of_bins=5):
        '''print((base_folder, event_folder, depth_folder, frame_folder, flow_folder, semantic_folder, \
                 start_time, stop_time, clip_distance, every_x_rgb_frame, \
                 transform, normalize, use_phased_arch, baseline))'''

        self.base_folder = base_folder
        self.depth_folder = join(self.base_folder, depth_folder if depth_folder is not None else 'frames')
        self.frame_folder = join(self.base_folder, frame_folder if frame_folder is not None else 'rgb')
        self.flow_folder = join(self.base_folder, flow_folder if flow_folder is not None else 'flow')
        self.semantic_folder = join(self.base_folder, semantic_folder if semantic_folder is not None else 'semantic')
        self.transform = transform
        self.normalize = normalize
        self.event_dataset = RawEventsDataset(base_folder, event_folder, start_time, stop_time)
        self.eps = 1e-06
        self.clip_distance = clip_distance
        self.use_phased_arch = use_phased_arch
        self.every_x_rgb_frame = every_x_rgb_frame
        self.baseline = baseline
        self.loss_composition = loss_composition
        self.nbr_of_events_per_voxelgrid = nbr_of_events_per_voxelgrid
        self.nbr_of_bins = nbr_of_bins

        self.debug = True

        # get dimensions by loading dummy_frame
        #dummy_frame = np.load(join(self.depth_folder, 'depth_0000000000.npy')).astype(np.float32)
        path_dummy_depthframe = glob.glob(self.depth_folder + '/*_0000_depth.npy')
        dummy_frame = np.load(path_dummy_depthframe[0]).astype(np.float32)
        self.height, self.width = dummy_frame.shape

        # create dummy input to know shape of voxelgrids after transform
        dummy_input = self.events_to_voxel_grid(np.zeros([10, 4]), self.nbr_of_bins, self.height, self.width)
        dummy_input = self.transform(torch.from_numpy(dummy_input))
        self.height_voxelgrid, self.width_voxelgrid = dummy_input.shape[1], dummy_input.shape[2]

        # Load the stamp files
        self.stamps = np.loadtxt(
            join(self.depth_folder, 'timestamps.txt'))[:, 1]

        # shift the frame timestamps by the same amount as the event timestamps
        self.stamps -= self.event_dataset.initial_stamp

        # length = total number of datapoints, not equal to length of final dataset due to every_x_rgb_frame & step size
        self.length = len(self.event_dataset)

        # Check that the frame timestamps are unique and sorted
        assert(np.alltrue(np.diff(self.stamps) > 0)
               ), "frame timestamps are not unique and monotonically increasing"

        # Check that the latest frame in the dataset has a timestamp >= the latest event frame
        assert(
            self.stamps[-1] >= self.event_dataset.get_last_stamp())

        self.nbr_events_per_frame = []




    def __len__(self):
        return self.length

    def __getitem__(self, i, seed=None, reg_factor=5.70378):
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

        events_overall = None
        rgb_overall = []

        for k in range(0, self.every_x_rgb_frame):
            j = i * self.every_x_rgb_frame + k
            event_timestamp = self.event_dataset.get_stamp_at(j)

            # Find the index of the first frame whose timestamp is >= event timestamp
            (frame_idx, frame_timestamp) = first_element_greater_than(self.stamps, event_timestamp)
            assert(frame_idx >= 0)
            assert(frame_idx < len(self.stamps))
            # assert(frame_timestamp >= event_timestamp)
            assert(frame_timestamp == event_timestamp)
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
            events = self.event_dataset.__getitem__(j, seed)
            if events_overall is None:
                events_overall = events
            else:
                events_overall = np.append(events_overall, events, axis=0)

        # construct voxelgrids out of raw events
        total_events = events_overall.shape[0]

        nbr_voxelgrids = 1
        nbr_events_per_voxelgrid = int(np.floor(total_events / nbr_voxelgrids))
        # print(nbr_voxelgrids, nbr_events_per_voxelgrid, total_events)

        if self.baseline == 'ergb':
            all_voxelgrids = torch.zeros(nbr_voxelgrids, self.nbr_of_bins+1, self.height_voxelgrid, self.width_voxelgrid)
        else:
            all_voxelgrids = torch.zeros(nbr_voxelgrids, self.nbr_of_bins, self.height_voxelgrid, self.width_voxelgrid)

        voxelgrid = self.events_to_voxel_grid(events_overall, self.nbr_of_bins, self.height, self.width)

        # for debugging:
        # print("mean: ", [np.mean(voxelgrid[ii, :, :]) for ii in range(self.nbr_of_bins)])
        # print("max: ", [np.max(voxelgrid[ii, :, :]) for ii in range(self.nbr_of_bins)])
        # print("min: ", [np.min(voxelgrid[ii, :, :]) for ii in range(self.nbr_of_bins)])

        if self.normalize:
            voxelgrid = self.normalize_voxelgrid(voxelgrid)
        voxelgrid = torch.from_numpy(voxelgrid)
        if self.transform:
            random.seed(seed)
            voxelgrid = self.transform(voxelgrid)

        # Get RGB frame and corresponding depth frame
        # Load numpy depth ground truth depth frame
        # frame = np.load(join(self.depth_folder, 'depth_{:010d}.npy'.format(frame_idx))).astype(np.float32)
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
        # frame = frame / np.amax(frame[~np.isnan(frame)])
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

        if self.use_phased_arch:
            timestamp = torch.from_numpy(np.asarray([event_timestamp]).astype(np.float32))
            # timestamp = torch.tensor([event_timestamp])

        try:
            # rgb_frame = io.imread(join(self.frame_folder, 'frame_{:010d}.png'.format(frame_idx)),
            #                      as_gray=False).astype(np.float32)
            path_rgbframe = glob.glob(self.frame_folder + '/*_{:04d}_image.png'.format(frame_idx))
            rgb_frame = io.imread(path_rgbframe[0], as_gray=False).astype(np.float32)
            if rgb_frame.shape[2] > 1:
                gray_frame = rgb2gray(rgb_frame)  # [H x W]

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

        item['image'] = torch.cat((voxelgrid, gray_frame), axis=0)
        item['depth_image'] = frame
        if self.use_phased_arch:
            item['times_image'] = timestamp

        return item

    def events_to_voxel_grid(self, events, num_bins, height, width):
        """
        Build a voxel grid with bilinear interpolation in the time domain from a set of events.
        :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
        :param num_bins: number of bins in the temporal axis of the voxel grid
        :param width, height: dimensions of the voxel grid
        """

        assert (events.shape[1] == 4)
        assert (num_bins > 0)
        assert (width > 0)
        assert (height > 0)

        voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

        # normalize the event timestamps so that they lie between 0 and num_bins
        last_stamp = events[-1, 0]
        first_stamp = events[0, 0]
        deltaT = last_stamp - first_stamp

        if deltaT == 0:
            deltaT = 1.0

        events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT
        ts = events[:, 0]
        xs = events[:, 1].astype(np.int)
        ys = events[:, 2].astype(np.int)
        pols = events[:, 3]
        pols[pols == 0] = -1  # polarity should be +1 / -1

        tis = ts.astype(np.int)
        dts = ts - tis
        vals_left = pols * (1.0 - dts)
        vals_right = pols * dts

        valid_indices = tis < num_bins
        np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width +
                  tis[valid_indices] * width * height, vals_left[valid_indices])

        valid_indices = (tis + 1) < num_bins
        np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width +
                  (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

        voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))

        return voxel_grid

    def normalize_voxelgrid(self, event_tensor):
        # normalize the event tensor (voxel grid) in such a way that the mean and stddev of the nonzero values
        # in the tensor are equal to (0.0, 1.0)
        mask = np.nonzero(event_tensor)
        if mask[0].size > 0:
            mean, stddev = event_tensor[mask].mean(), event_tensor[mask].std()
            if stddev > 0:
                event_tensor[mask] = (event_tensor[mask] - mean) / stddev
        return event_tensor