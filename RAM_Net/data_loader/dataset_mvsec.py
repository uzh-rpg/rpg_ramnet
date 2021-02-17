# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as f

import pprint
import glob
from numpy.lib.format import open_memmap
import cv2
import tqdm
import matplotlib.pyplot as plt
import random
from .dataset_asynchronous import normalize_voxelgrid
from .dataset_ddd17 import DDD17SegmentationBase
from utils.event_tensor_utils import events_to_voxel_grid
from torch.utils.data.dataloader import default_collate
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


class SequenceMVSEC(Dataset):
    """Load sequences of time-synchronized {event tensors + depth} from a folder."""

    def __init__(self, base_folder, event_folder="events", depth_folder='frames', frame_folder='rgb', flow_folder='flow',
                 semantic_folder='semantic', start_time=0.0, stop_time=0.0, sequence_length=2, transform=None,
                 proba_pause_when_running=0.0, proba_pause_when_paused=0.0, step_size=20, clip_distance=100.0,
                 normalize=True, scale_factor=1.0, use_phased_arch=False, every_x_rgb_frame=1, baseline=False,
                 loss_composition=False, event_loader="voxelgrids", nbr_of_events_per_voxelgrid=0, nbr_of_bins=5):

        assert(sequence_length > 0)
        assert(step_size > 0)
        assert(clip_distance > 0)
        self.L = sequence_length
        if baseline == "rgb":
            self.dataset = MVSEC_Images(base_folder, clip_distance=clip_distance, transform=transform,
                                        normalize=normalize)
        else:
            self.dataset = MVSEC_Data(base_folder, clip_distance=clip_distance, transform=transform,
                                      normalize=normalize, nbr_of_events_per_voxelgrid=nbr_of_events_per_voxelgrid,
                                      nbr_of_bins=nbr_of_bins)
        self.step_size = step_size
        self.every_x_rgb_frame = every_x_rgb_frame
        if self.L >= self.dataset.length:
            self.length = 0
        else:
            self.length = (self.dataset.length - self.L) // self.step_size + 1
        print("lenght sequence dataset: ", self.length)

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

        sequence = []

        # add the first element (i.e. do not start with a pause)
        k = 0
        j = i * self.step_size
        item = self.dataset.__getitem__(j, seed)
        sequence.append(item)

        for n in range(self.L - 1):
            k += 1
            item = self.dataset.__getitem__(j + k, seed)
            sequence.append(item)
        #print(sum(len(k) for k in sequence))
        # down sample data
        if self.scale_factor < 1.0:
            for data_items in sequence:
                for data_packages in data_items:
                    for k, item in data_packages.items():
                        if "events" in k:
                            item = item[None]
                            item = f.interpolate(item, scale_factor=self.scale_factor, mode='bilinear',
                                                 recompute_scale_factor=False, align_corners=False)
                            data_packages[k] = item[0]
                        elif k is not "img_loss":
                            item = item[None]
                            item = f.interpolate(item, scale_factor=self.scale_factor, mode='bilinear',
                                                 recompute_scale_factor=False, align_corners=False)
                            data_packages[k] = item[0]

        return sequence


class MVSEC_Data(Dataset):
    def __init__(self, base_folder,
                 clip_distance=100.0,
                 transform=None,
                 normalize=True,
                 nbr_of_events_per_voxelgrid=5000,
                 nbr_of_bins=5):

        self.base_folder = base_folder
        self.transform = transform
        self.normalize = normalize
        self.eps = 1e-06
        self.clip_distance = clip_distance
        self.nbr_of_events_per_voxelgrid = nbr_of_events_per_voxelgrid
        self.nbr_of_bins = nbr_of_bins

        self.fill_out_nans = False

        if "ddd17" in base_folder:
            self.dataset_raw = DDD17SegmentationBase(base_folder)
            self.length = len(self.dataset_raw)
            print("lenght ddd17 dataset: ", self.length)
            # get dimensions by loading dummy_frame
            dummy_frame = self.dataset_raw[0][-1]["frame"]
            print(dummy_frame.shape)
            self.height, self.width, _ = dummy_frame.shape
        else:
            self.dataset_raw = MVSEC_Raw_Data(base_folder)
            self.length = len(self.dataset_raw)
            print("lenght mvsec dataset: ", self.length)
            # get dimensions by loading dummy_depth
            dummy_depth = self.dataset_raw[0][-1]["depth"]
            self.height, self.width = dummy_depth.shape

        # create dummy input to know shape of voxelgrids after transform
        dummy_input = events_to_voxel_grid(np.zeros([10, 4]), self.nbr_of_bins, self.width, self.height)
        dummy_input = self.transform(torch.from_numpy(dummy_input))
        self.height_voxelgrid, self.width_voxelgrid = dummy_input.shape[1], dummy_input.shape[2]

    def __len__(self):
        return self.length

    def prepare_depth(self, depth, seed, reg_factor):
        # Clip to maximum distance
        depth = np.clip(depth, 0.0, self.clip_distance)
        # Normalize
        depth = depth / self.clip_distance
        # Convert to log depth
        depth = 1.0 + np.log(depth) / reg_factor
        # Clip between 0 and 1.0
        depth = depth.clip(0, 1.0)

        if len(depth.shape) == 2:  # [H x W] grayscale image -> [H x W x 1]
            depth = np.expand_dims(depth, -1)

        depth = np.moveaxis(depth, -1, 0)  # H x W x C -> C x H x W

        if self.fill_out_nans:
            upper, lower = np.vsplit(depth[0], 2)
            upper = np.nan_to_num(upper, nan=1.0)
            # lower = np.nan_to_num(lower, nan=0.0)
            depth = np.vstack([upper, lower])
            depth = depth[None, :]

        depth = torch.from_numpy(depth)  # numpy to tensor

        if self.transform:
            random.seed(seed)
            depth = self.transform(depth)
        return depth

    def rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.float32)

    def prepare_frame(self, frame, seed):
        if len(frame.shape) > 2:
            if frame.shape[2] > 1:
                frame = self.rgb2gray(frame)  # [H x W]

        frame /= 255.0  # normalize
        frame = np.expand_dims(frame, axis=0)  # expand to [1 x H x W]
        frame = torch.from_numpy(frame)
        if self.transform:
            random.seed(seed)
            frame = self.transform(frame)
        return frame

    def __getitem__(self, i, seed=None, reg_factor=3.70378):
        """
                This function returns a tuple (depth, data) where depth is the label and data has the following form:
                data = [
                    {
                        "events": [[...]] events between first depth and first image
                    },
                    {
                        "frame": [[...]]  first image
                    }
                    ...
                    {
                        "events": [[...]] events between frame i and frame i+1
                    },
                    {
                        "frame": [[...]]  frame i
                    },
                    ...
                    {
                        "events": [[...]] events between last image and last depth
                    }
                ]

                Here events have the format (x,y,t,p) columns, p\in {0,1} and t is in seconds (float64)

                Voxelgrids are constructed such that each voxelgrid contains at least
                self.nbr_events_per_voxelgrid * 0.5 events.
                """

        data = self.dataset_raw.__getitem__(i)
        item = []
        assert(i >= 0)
        # assert(i < (self.length // self.every_x_rgb_frame))

        if seed is None:
            # if no specific random seed was passed, generate our own.
            # otherwise, use the seed that was passed to us
            seed = random.randint(0, 2 ** 32)

        event_index = 0
        frame_index = 0

        overflow_events = None  # used if an event package doesn't contain enough events for a big enough voxelgrid

        nbr_data_enrtries = len(data)
        index_helper = 0
        second_last_event_index = None
        for k, entry in reversed(list(enumerate(data))):
            if "events" in list(entry.keys())[0] and index_helper == 0:
                last_event_index = k
                index_helper += 1
            elif "events" in list(entry.keys())[0]:
                second_last_event_index = k
                break
        # print("last indexes: ", last_event_index, second_last_event_index)
        # print("keys: ", [list(data_entry.keys())[0] for data_entry in data])

        for j, data_item in enumerate(data):
            if "events" in data_item.keys():
                if overflow_events is not None:
                    events_overall = np.append(overflow_events, data_item['events'], axis=0)
                    overflow_events = None
                else:
                    events_overall = data_item['events']

                total_events = events_overall.shape[0]
                #print("total events before: ", total_events)
                # check if this is the second last event package and last event package is too small for whole voxelgrid
                # (for nbr_events < 50% self.nbr_of_events_per_voxelgrid, if this is the case, the two last event
                # packages are appended to get a voxelgrid that is large enough for a good prediction for the loss)
                if j == second_last_event_index:
                    total_events_last_package = data[last_event_index]['events'].shape[0]
                    if total_events_last_package < self.nbr_of_events_per_voxelgrid * 0.5:
                        if total_events < self.nbr_of_events_per_voxelgrid:
                            overflow_events = events_overall
                            continue
                        else:
                            overflow_events = events_overall[-int(self.nbr_of_events_per_voxelgrid*0.5):, :]
                            events_overall = events_overall[0:-int(self.nbr_of_events_per_voxelgrid*0.5), :]

                # construct voxelgrids out of raw events
                total_events = events_overall.shape[0]
                #print("total events after: ", total_events)
                if total_events < self.nbr_of_events_per_voxelgrid * 0.5 and j != last_event_index:
                    # if this event package doesn't contain enough events for a voxelgrid, it is concatenated
                    # with the next event package.
                    overflow_events = events_overall
                    continue
                elif total_events < self.nbr_of_events_per_voxelgrid:
                    nbr_voxelgrids = 1
                else:
                    nbr_voxelgrids = int(np.floor(total_events / self.nbr_of_events_per_voxelgrid))
                nbr_voxelgrids = 10 if nbr_voxelgrids > 10 else nbr_voxelgrids
                nbr_events_per_voxelgrid = int(np.floor(total_events / nbr_voxelgrids))
                #print("nbr voxelgrids: ", nbr_voxelgrids, nbr_events_per_voxelgrid)
                for i in range(nbr_voxelgrids):
                    # if total_events < (i + 2) * nbr_events_per_voxelgrid:
                    if i == nbr_voxelgrids - 1:
                        # if this is the last voxelgrid of this datapack, use the rest of the data.
                        voxelgrid = events_to_voxel_grid(
                            events_overall[i * nbr_events_per_voxelgrid:-1], self.nbr_of_bins, self.width, self.height)
                    else:
                        voxelgrid = events_to_voxel_grid(
                            events_overall[i * nbr_events_per_voxelgrid:(i + 1) * nbr_events_per_voxelgrid],
                            self.nbr_of_bins, self.width, self.height)
                    # for debugging:
                    # print("mean: ", [np.mean(voxelgrid[ii, :, :]) for ii in range(self.nbr_of_bins)])
                    # print("max: ", [np.max(voxelgrid[ii, :, :]) for ii in range(self.nbr_of_bins)])
                    # print("min: ", [np.min(voxelgrid[ii, :, :]) for ii in range(self.nbr_of_bins)])

                    #fig, ax = plt.subplots(ncols=1, nrows=1)
                    #ax.imshow(np.sum(voxelgrid, axis=0))
                    #ax.set_title(("events"))
                    #plt.show()

                    if self.normalize:
                        voxelgrid = normalize_voxelgrid(voxelgrid)
                    voxelgrid = torch.from_numpy(voxelgrid)
                    if self.transform:
                        random.seed(seed)
                        voxelgrid = self.transform(voxelgrid)

                    item += [{"events{}".format(event_index): voxelgrid}]
                    event_index += 1

            elif "frame" in data_item.keys():
                frame = data_item['frame']
                frame = self.prepare_frame(frame, seed)
                item += [{"image{}".format(frame_index): frame}]
                frame_index += 1

            elif "depth" in data_item.keys():
                depth = data_item['depth']
                depth = self.prepare_depth(depth, seed, reg_factor)
                item += [{"depth": depth}]

        timestamp_depth = data[-1]["timestamp"]
        if len(data) > 2:
            # if data doesnt contain an image, it's length is 2 (events, depth)
            timestamp_image = data[-3]["timestamp"]
            if timestamp_depth - timestamp_image < 0.2 * 0.05:
                # image loss is only calculated if last image was seen less than 20% of depth frame rate ago.
                flag_image_loss = True
            else:
                flag_image_loss = False
        else:
            flag_image_loss = False

        item += [{"img_loss": flag_image_loss}]
        #print("keys: ", [list(item_entry.keys())[0] for item_entry in item])
        return item


class MVSEC_Raw_Data(Dataset):
    def __init__(self, base_folder):

        self.base_folder = base_folder

        self.target_label = "depth"
        self.events_label = "events-betweenframes"
        self.frame_label = "frames"

        self.depth_valid_indices = {}
        self.depth_to_event_index = {}
        self.depth_to_frame_index = {}
        self.frame_to_event_index = {}
        
        self.depth_paths = {}
        self.frame_paths = {}
        self.event_paths = {}
        self.event_handles = {}

        self.depth_timestamps = {}
        self.frame_timestamps = {}

        for path, subdirs, _ in os.walk(self.base_folder):
            if self.target_label not in subdirs:
                continue 
            print(path) 
            valid = np.load(os.path.join(path, "valid_depth_indices.npy"))
            self.depth_to_event_index[path] = np.load(os.path.join(path, "depth_to_event_index.npy"))
            self.depth_to_frame_index[path] = np.load(os.path.join(path, "depth_to_frame_index.npy"))
            self.depth_timestamps[path] = np.genfromtxt(os.path.join(path, self.target_label, "timestamps.txt"), dtype="float64")[valid,1]
            self.frame_timestamps[path] = np.genfromtxt(os.path.join(path, self.frame_label, "timestamps.txt"), dtype="float64")[:,1]

            self.frame_to_event_index[path] = np.load(os.path.join(path, "frame_to_event_index.npy"))

            self.depth_paths[path] = sorted(glob.glob(os.path.join(path, self.target_label,"*.npy")))
            self.frame_paths[path] = sorted(glob.glob(os.path.join(path, self.frame_label,"*.png")))
            #self.event_paths[path] = sorted(glob.glob(os.path.join(path, self.events_label,"*.npy")))
            self.event_handles[path] = [
                open_memmap(os.path.join(path, self.events_label, "t.npy"), mode='r'),
                open_memmap(os.path.join(path, self.events_label, "xy.npy") , mode='r'),
                open_memmap(os.path.join(path, self.events_label, "p.npy") , mode='r')
            ]

        self.roots = sorted(list(self.event_handles.keys()))

    def __len__(self):
        l = 0
        for path in self.roots:
            l += (len(self.depth_to_event_index[path])-1)
        return l

    def get_rel_index(self, item):
        for k in self.roots:
            num_indices = (len(self.depth_to_event_index[k])-1)
            if num_indices > item:
                return item, k
            item -= num_indices
        else:
            raise ValueError

    def __getitem__(self, item):
        """
        This function returns a tuple (depth, data) where depth is the label and data has the following form:
        data = [
            {
                "events": [[...]] events between first depth and first image
            },
            {
                "frame": [[...]]  first image
            }
            ...
            {
                "events": [[...]] events between frame i and frame i+1
            },
            {
                "frame": [[...]]  frame i
            },
            ...
            {
                "events": [[...]] events between last image and last depth
            }
        ]

        Here events have the format (x,y,t,p) columns, p\in {0,1} and t is in seconds (float64)
        """

        relative_index, root = self.get_rel_index(item)
        
        # indices
        depth_to_event_index = self.depth_to_event_index[root]
        frame_to_event_index = self.frame_to_event_index[root]
        depth_to_frame_index = self.depth_to_frame_index[root]

        # timestamps 
        depth_timestamps = self.depth_timestamps[root]
        frame_timestamps = self.frame_timestamps[root]

        # events
        event_t, event_xy, event_p = self.event_handles[root]

        # get events and images
        data = []
        frame_index_max = depth_to_frame_index[relative_index+1]
        frame_index_min = depth_to_frame_index[relative_index]

        # first put all events between d_min and f_min and frame at f_min
        if frame_index_max == frame_index_min:
            # if there is no frame in this data item, events_max_index corresponds to depth index.
            event_max_index = depth_to_event_index[relative_index+1]
        else:
            event_max_index = frame_to_event_index[frame_index_min]
        event_min_index = depth_to_event_index[relative_index]

        events = np.concatenate([
            event_t[event_min_index:event_max_index,:].astype("float64"),
            event_xy[event_min_index:event_max_index,:].astype("float32"),
            (event_p[event_min_index:event_max_index,:].astype("float32")*2-1)
        ], axis=-1)

        data += [{"events": events}]

        # if there is no frame in this data item, frame_index_max == frame_index_min, therefore no further
        # events or images have to be added.
        if frame_index_max != frame_index_min:
            frame = cv2.imread(self.frame_paths[root][frame_index_min])
            data += [{"frame": frame, "timestamp": frame_timestamps[frame_index_min]}]

            # get all events between images and add images until last image
            for frame_index in range(frame_index_min, frame_index_max-1):
                event_max_index = frame_to_event_index[frame_index+1]
                event_min_index = frame_to_event_index[frame_index]

                frame = cv2.imread(self.frame_paths[root][frame_index+1])

                events = np.concatenate([
                    event_t[event_min_index:event_max_index,:].astype("float64"),
                    event_xy[event_min_index:event_max_index,:].astype("float32"),
                    (event_p[event_min_index:event_max_index,:].astype("float32")*2-1)
                ], axis=-1)

                data += [{"events": events}, {"frame": frame, "timestamp": frame_timestamps[frame_index+1]}]

            # get all events between f_max and d_max
            event_min_index = frame_to_event_index[frame_index_max-1]
            event_max_index = depth_to_event_index[relative_index+1]

            events = np.concatenate([
                event_t[event_min_index:event_max_index,:].astype("float64"),
                event_xy[event_min_index:event_max_index,:].astype("float32"),
                (event_p[event_min_index:event_max_index,:].astype("float32")*2-1)
            ], axis=-1)

            data += [{"events": events}]

        # get label
        label = np.load(os.path.join(self.depth_paths[root][relative_index+1]))
        data += [{"depth": label, "timestamp": depth_timestamps[relative_index+1]}]

        #for e in data:
        #   print({k:type(v) for k, v in e.items()})
        return data


class MVSEC_Images(Dataset):
    def __init__(self, base_folder,
                 clip_distance=100.0,
                 transform=None,
                 normalize=True,
                 baseline=False):

        self.base_folder = base_folder
        self.transform = transform
        self.normalize = normalize
        self.clip_distance = clip_distance
        self.baseline = baseline

        self.target_label = "depth"
        self.frame_label = "frames"

        self.depth_valid_indices = {}
        self.depth_to_frame_index = {}

        self.depth_paths = {}
        self.frame_paths = {}

        self.depth_timestamps = {}
        self.frame_timestamps = {}

        self.fill_out_nans = False


        for path, subdirs, _ in os.walk(self.base_folder):
            if self.target_label not in subdirs:
                continue
            print(path)
            valid = np.load(os.path.join(path, "valid_depth_indices.npy"))
            self.depth_to_frame_index[path] = np.load(os.path.join(path, "depth_to_frame_index.npy"))
            self.depth_timestamps[path] = \
            np.genfromtxt(os.path.join(path, self.target_label, "timestamps.txt"), dtype="float64")[valid, 1]
            self.frame_timestamps[path] = np.genfromtxt(os.path.join(path, self.frame_label, "timestamps.txt"),
                                                        dtype="float64")[:, 1]

            self.depth_paths[path] = sorted(glob.glob(os.path.join(path, self.target_label, "*.npy")))
            self.frame_paths[path] = sorted(glob.glob(os.path.join(path, self.frame_label, "*.png")))

        self.roots = sorted(list(self.frame_paths.keys()))

        # length of dataset:
        length = 0
        for path in self.roots:
            length += (len(self.depth_to_frame_index[path]) - 1)
        self.length = length

    def __len__(self):
        return self.length

    def get_rel_index(self, item):
        for k in self.roots:
            num_indices = (len(self.depth_to_frame_index[k]) - 1)
            if num_indices > item:
                return item, k
            item -= num_indices
        else:
            raise ValueError

    def interpolate_frames(self, frame_before, frame_after, t_0, t, t_1):
        frame = (frame_after - frame_before) * (t - t_0) / (t_1 - t_0) + frame_before
        return frame

    def prepare_depth(self, depth, seed, reg_factor):
        # Clip to maximum distance
        depth = np.clip(depth, 0.0, self.clip_distance)
        # Normalize
        depth = depth / self.clip_distance
        # Convert to log depth
        depth[depth == 0] = 1e-6
        depth = 1.0 + np.log(depth) / reg_factor
        # Clip between 0 and 1.0
        depth = depth.clip(0, 1.0)

        if len(depth.shape) == 2:  # [H x W] grayscale image -> [H x W x 1]
            depth = np.expand_dims(depth, -1)

        depth = np.moveaxis(depth, -1, 0)  # H x W x C -> C x H x W

        if self.fill_out_nans:
            upper, lower = np.vsplit(depth[0], 2)
            upper = np.nan_to_num(upper, nan=1.0)
            # lower = np.nan_to_num(lower, nan=0.0)
            depth = np.vstack([upper, lower])
            depth = depth[None, :]

        depth = torch.from_numpy(depth)  # numpy to tensor

        if self.transform:
            random.seed(seed)
            depth = self.transform(depth)
        return depth

    def rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.float32)

    def prepare_frame(self, frame, seed):
        if len(frame.shape) > 2:
            if frame.shape[2] > 1:
                frame = self.rgb2gray(frame)  # [H x W]

        frame /= 255.0  # normalize
        frame = np.expand_dims(frame, axis=0)  # expand to [1 x H x W]
        frame = torch.from_numpy(frame)
        if self.transform:
            random.seed(seed)
            frame = self.transform(frame)
        return frame

    def __getitem__(self, item, seed=None, reg_factor=3.70378):
        """
        This function returns a list of data in the following form:
        data = [
            [{
                "image_0": [[...]]  first image, "timestamp": corresponding timestamp
            }]
            ...
            [{
                "image_X": [[...]]  last image, "timestamp": corresponding timestamp
            }],
            [{
                "depth": [[...]] label corresponding to last image
            }]
            [{
                "img_loss": bool added for consistency with events dataset
            }]

        ]
        """

        relative_index, root = self.get_rel_index(item)

        # indices
        depth_to_frame_index = self.depth_to_frame_index[root]

        # timestamps
        depth_timestamps = self.depth_timestamps[root]
        frame_timestamps = self.frame_timestamps[root]

        image_index = 0

        # get images
        data = []
        frame_index_max = depth_to_frame_index[relative_index + 1]
        frame_index_min = depth_to_frame_index[relative_index]

        # if there is no frame in this data item, frame_index_max == frame_index_min, therefore no further
        # events or images have to be added.
        if frame_index_max != frame_index_min:
            frame = cv2.imread(self.frame_paths[root][frame_index_min])
            frame = self.prepare_frame(frame, seed)
            data += [{"image{}".format(image_index): frame, "timestamp": frame_timestamps[frame_index_min]}]
            image_index += 1

            # get all events between images and add images until last image
            for frame_index in range(frame_index_min, frame_index_max - 1):
                frame = cv2.imread(self.frame_paths[root][frame_index + 1])
                frame = self.prepare_frame(frame, seed)

                data += [{"image{}".format(image_index): frame, "timestamp": frame_timestamps[frame_index + 1]}]
                image_index += 1

            # get time synchronized label by interpolating two labels
            label_before = np.load(os.path.join(self.depth_paths[root][relative_index]))
            label_after = np.load(os.path.join(self.depth_paths[root][relative_index + 1]))
            # print(np.min(label_after[~np.isnan(label_after)]), np.max(label_after[~np.isnan(label_after)]))

            label_before_ts = depth_timestamps[relative_index]
            label_after_ts = depth_timestamps[relative_index + 1]
            image_ts = data[-1]["timestamp"]
            label = self.interpolate_frames(label_before, label_after, label_before_ts, image_ts, label_after_ts)
            label = self.prepare_depth(label, seed, reg_factor)
            data += [{"depth": label, "timestamp": depth_timestamps[relative_index + 1]}]

        else:
            label = np.load(os.path.join(self.depth_paths[root][relative_index + 1]))
            label = self.prepare_depth(label, seed, reg_factor)
            data += [{"depth": label, "timestamp": depth_timestamps[relative_index + 1]}]

        data += [{"img_loss": True}]  # needs to be added to match dataset with events.

        #for e in data:
        #    print({k: type(v) for k, v in e.items()})
        return data


def collate_mvsec(batch):
    # batch: list of length batch_size. Each entry is a sequence of dicts, that contains the data
    # (keys: imageX, eventsX, depth, image_loss). size of batch: batch_size, seq_length, dict_entries, [1, x, x, x]
    # return_sequence should be a sequence of dicts, where each dict contains all corresponding
    # entries of the whole batch. size of return sequence: seq_length, dict entries, [batch_size, x, x, x]
    return_sequence = []
    sequence_length = len(batch[0])
    batch_size = len(batch)
    for j in range(sequence_length):
        # loop over the whole sequence to fill return_sequence list
        return_sequence += [[batch[i][j] for i in range(batch_size)]]
    return return_sequence




if __name__ == "__main__":
    #d = MVSEC_Raw_Data("/data/storage/michelle/mvsec/mvsec_daniel/outdoor_night1_data/")
    d = MVSEC_Data("/data/storage/michelle/mvsec/mvsec_daniel/outdoor_day1_data/",
                   transform=Compose([RandomRotationFlip(0.0, 0.5, 0.0), RandomCrop(224)]),
                   normalize=True)
    for i in range(len(d)):
        element = d[i]
        print(element[-1]['img_loss'])



