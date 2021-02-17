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
from utils.event_tensor_utils import events_to_voxel_grid
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
        dummy_input = events_to_voxel_grid(np.zeros([10, 4]), self.nbr_of_bins, self.height, self.width)
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

    def __getitem__(self, i, seed=None, reg_factor=3.70378):
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
            if self.baseline != 'rgb':
                events = self.event_dataset.__getitem__(j, seed)
            if events_overall is None:
                events_overall = events
            else:
                events_overall = np.append(events_overall, events, axis=0)
            print(events_overall)

            if self.baseline == "ergb":
                # get last seen rgb frame and add it to event tensors
                if frame_idx == 0 and k == 0:
                    path_dummy_input = glob.glob(self.frame_folder + '/*_0000_image.png')
                    dummy_input = io.imread(path_dummy_input[0], as_gray=False).astype(np.float32)
                    dummy_input = rgb2gray(dummy_input)  # [H x W]
                    dummy_input = np.expand_dims(dummy_input, axis=0)  # expand to [1 x H x W]
                    dummy_input = torch.from_numpy(dummy_input)
                    if self.transform:
                        random.seed(seed)
                        dummy_input = self.transform(dummy_input)
                    last_gray_frame = torch.zeros_like(dummy_input)
                elif k == 0 or k == self.every_x_rgb_frame - 1:
                    if k == 0:
                        # take the last seen rgb frame and copy it for all ergb tensors
                        path_rgbframe = glob.glob(self.frame_folder + '/*_{:04d}_image.png'.format(frame_idx-(k+1)))
                        rgb_frame = io.imread(path_rgbframe[0], as_gray=False).astype(np.float32)
                    elif k == self.every_x_rgb_frame - 1:
                        # for the last event in the batch, take the new/current rgb frame.
                        path_rgbframe = glob.glob(self.frame_folder + '/*_{:04d}_image.png'.format(frame_idx))
                        rgb_frame = io.imread(path_rgbframe[0], as_gray=False).astype(np.float32)
                    if rgb_frame.shape[2] > 1:
                        last_gray_frame = rgb2gray(rgb_frame)  # [H x W]
                    last_gray_frame /= 255.0  # normalize
                    last_gray_frame = np.expand_dims(last_gray_frame, axis=0)  # expand to [1 x H x W]
                    last_gray_frame = torch.from_numpy(last_gray_frame)
                    if self.transform:
                        random.seed(seed)
                        last_gray_frame = self.transform(last_gray_frame)

                    rgb_overall.append(last_gray_frame)

        # construct voxelgrids out of raw events
        total_events = events_overall.shape[0]
        # self.nbr_events_per_frame.append(total_events)
        if total_events < self.nbr_of_events_per_voxelgrid:
            nbr_voxelgrids = 1
        else:
            nbr_voxelgrids = int(np.floor(total_events / self.nbr_of_events_per_voxelgrid))
        nbr_voxelgrids = 10 if nbr_voxelgrids > 10 else nbr_voxelgrids
        nbr_events_per_voxelgrid = int(np.floor(total_events / nbr_voxelgrids))
        # print(nbr_voxelgrids, nbr_events_per_voxelgrid, total_events)

        if self.baseline == 'ergb':
            all_voxelgrids = torch.zeros(nbr_voxelgrids, self.nbr_of_bins+1, self.height_voxelgrid, self.width_voxelgrid)
        else:
            all_voxelgrids = torch.zeros(nbr_voxelgrids, self.nbr_of_bins, self.height_voxelgrid, self.width_voxelgrid)

        for i in range(nbr_voxelgrids):
            # if total_events < (i + 2) * nbr_events_per_voxelgrid:
            if i == nbr_voxelgrids - 1:
                # if this is the last voxelgrid of this datapack, use the rest of the data.
                voxelgrid = events_to_voxel_grid(
                    events_overall[i * nbr_events_per_voxelgrid:-1], self.nbr_of_bins, self.height, self.width)
            else:
                voxelgrid = events_to_voxel_grid(
                    events_overall[i * nbr_events_per_voxelgrid:(i + 1) * nbr_events_per_voxelgrid],
                    self.nbr_of_bins, self.height, self.width)
            # for debugging:
            # print("mean: ", [np.mean(voxelgrid[ii, :, :]) for ii in range(self.nbr_of_bins)])
            # print("max: ", [np.max(voxelgrid[ii, :, :]) for ii in range(self.nbr_of_bins)])
            # print("min: ", [np.min(voxelgrid[ii, :, :]) for ii in range(self.nbr_of_bins)])
            if self.normalize:
                voxelgrid = normalize_voxelgrid(voxelgrid)
            voxelgrid = torch.from_numpy(voxelgrid)
            if self.transform:
                random.seed(seed)
                voxelgrid = self.transform(voxelgrid)
            # print([torch.mean(voxelgrid[ii, :, :]) for ii in range(self.nbr_of_bins)])
            if self.baseline == 'ergb':
                if i < nbr_voxelgrids - 1:
                    all_voxelgrids[i, :, :, :] = torch.cat((voxelgrid, rgb_overall[0]), axis=0)
                elif i == nbr_voxelgrids - 1:
                    all_voxelgrids[i, :, :, :] = torch.cat((voxelgrid, rgb_overall[-1]), axis=0)
            else:
                all_voxelgrids[i, :, :, :] = voxelgrid

        if not bool(self.baseline) or self.baseline == 'e' or self.baseline == 'ergb':
            item['events'] = all_voxelgrids
            item['batchlength_events'] = nbr_voxelgrids

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

        if self.frame_folder is not None and (not bool(self.baseline) or self.baseline == 'rgb'):
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

        if not bool(self.baseline) or self.baseline == 'rgb':
            item['image'] = gray_frame
        item['depth_image'] = frame
        if self.use_phased_arch:
            item['times_image'] = timestamp

        return item


def normalize_voxelgrid(event_tensor):
        # normalize the event tensor (voxel grid) in such a way that the mean and stddev of the nonzero values
        # in the tensor are equal to (0.0, 1.0)
        mask = np.nonzero(event_tensor)
        if mask[0].size > 0:
            mean, stddev = event_tensor[mask].mean(), event_tensor[mask].std()
            if stddev > 0:
                event_tensor[mask] = (event_tensor[mask] - mean) / stddev
        return event_tensor


def my_collate(batch):
    # batch: list of length batch_size. Each entry is a sequence of dicts, that contains the data
    # (keys: image, events, depth_image). size of batch: batch_size, seq_length, dict_entries, [1, x, x, x]
    # return_sequence should be a sequence of dicts, where each dict contains all corresponding
    # entries of the whole batch. size of return sequence: seq_length, dict entries, [batch_size, x, x, x]
    print("preparing batch in collate")
    return_sequence = []
    sequence_length = len(batch[0])
    batch_size = len(batch)
    for j in range(sequence_length):
        # loop over the whole sequence to fill return_sequence list
        return_dict = {}  # entry of return_sequence
        for key in batch[0][j].keys():
            if key != "events":
                # all keys apart from "events" have the same dimension for each sample in the batch, therefore the
                # default_collate function can be used.
                return_dict[key] = default_collate([seq_item[key] for seq_item in [seq[j] for seq in batch]])
        max_nbr_voxelgrids = max([batch[i][j]['batchlength_events'] for i in range(batch_size)])
        # create zero tensor in a shape that is able to include the largest batch entry. All other entries will be
        # automatically padded with zeros.
        events = torch.zeros(batch_size, max_nbr_voxelgrids, batch[0][j]['events'].shape[1],
                             batch[0][j]['events'].shape[2], batch[0][j]['events'].shape[3])
        for i in range(batch_size):
            nbr_voxelgrids = batch[i][j]['batchlength_events']
            # print("nbr voxelgrids: ", nbr_voxelgrids)
            # print("shape input: ", batch[i][j]['events'].shape)
            events[i, 0:nbr_voxelgrids, :, :, :] = batch[i][j]['events']
        return_dict['events'] = events
        # print("shape events: ", events.shape)
        return_sequence.append(return_dict)

    return return_sequence
