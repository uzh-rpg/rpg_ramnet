import numpy as np
import glob 
import cv2
import tqdm
import os
from utils.event_tensor_utils import events_to_voxel_grid
from .dataset_asynchronous import normalize_voxelgrid
from torch.utils.data import Dataset



def load_files_in_directory(directory, dt_ms=50):
    # Load files: these include these have the following form
    #
    # idx.npy :
    #   t0_ns idx0
    #   t1_ns idx1
    #   ...
    #   tj_ns idxj
    #   ...
    #   tN_ns idxN
    #
    # This file contains a mapping from j -> tj_ns idxj,
    # where j+1 is the idx of the img with timestamp tj_ns (in nanoseconds)
    # and idxj is the idx of the last event before the img (in events.dat.t and events.dat.xyp)
    img_timestamp_event_idx = np.load(os.path.join(directory, "index", "index_%ims.npy" % dt_ms))
    # events.dat.t :
    #   t0_ns
    #   t1_ns
    #   ...
    #   tM_ns
    #
    #  events.dat.xyp :
    #    x0 y0 p0
    #    ...
    #    xM yM pM
    events_t_file = os.path.join(directory, "events.dat.t")
    events_xyp_file = os.path.join(directory, "events.dat.xyp")

    t_events, xyp_events = load_events(events_t_file, events_xyp_file)

    # Since the imgs are in a video format, they cannot be loaded directly, however, the segmentation masks from the
    # original dataset (EV-SegNet) have been copied into this folder. First unzip the segmentation masks with
    #
    #    unzip segmentation_masks.zip
    #
    img_files = sorted(glob.glob(os.path.join(directory, "imgs", "*.png")))

    return img_timestamp_event_idx, t_events, xyp_events, img_files


def load_events(t_file, xyp_file):
    # events.dat.t saves the timestamps of the indiviual events (in nanoseconds -> int64)
    # events.dat.xyp saves the x, y and polarity of events in uint8 to save storage. The polarity is 0 or 1.
    # We first need to compute the number of events in the memmap since it does not do it for us. We can do
    # this by computing the file size of the timestamps and dividing by 8 (since timestamps take 8 bytes)

    num_events = int(os.path.getsize(t_file) / 8)
    t_events = np.memmap(t_file, dtype="int64", mode="r", shape=(num_events, 1))
    xyp_events = np.memmap(xyp_file, dtype="int16", mode="r", shape=(num_events, 3))

    return t_events, xyp_events


def extract_events_from_memmap(t_events, xyp_events, img_idx, img_timestamp_event_idx):
    _, event_idx_before, _ = img_timestamp_event_idx[img_idx]
    _, event_idx, _ = img_timestamp_event_idx[img_idx+1]
    
    event_idx_before = max([event_idx_before, 0])
    event_idx = min([event_idx, len(t_events)-2])

    '''events = np.concatenate([
        event_t[event_min_index:event_max_index, :].astype("float64"),
        event_xy[event_min_index:event_max_index, :].astype("float32"),
        (event_p[event_min_index:event_max_index, :].astype("float32") * 2 - 1)
    ], axis=-1)'''

    events_between_imgs = np.concatenate([
        np.asarray(t_events[event_idx_before+1:event_idx+1], dtype="int64"),
        np.asarray(xyp_events[event_idx_before+1:event_idx+1], dtype="int64")
    ], -1)
    # events_between_imgs = events_between_imgs[:, [1, 2, 0, 3]]  # events have format xytp, and p is in [0,1]
    events_between_imgs[:, 3] = events_between_imgs[:, 3] * 2 - 1  # events have format tpxy, and p is in [-1,1]
    return events_between_imgs



class DDD17SegmentationBase(Dataset):
    def __init__(self, root, 
                       num_ms_events=50):
        print(root)
  
        self.img_timestamp_event_idx = {}
        self.event_data = {}
        self.img_files = {}

        # for d in sorted(glob.glob(os.path.join(root, "dir*"))):
        for d in sorted(glob.glob(root)):
            img_timestamp_event_idx, t_events, xyp_events, img_files = load_files_in_directory(d, dt_ms=num_ms_events)
            self.img_timestamp_event_idx[d] = img_timestamp_event_idx
            self.event_data[d] = [t_events, xyp_events]
            self.img_files[d] = img_files

        self.roots = sorted(list(self.img_files.keys()))
        print(self.roots)

    def __len__(self):
        return sum(len(files)-1 for files in self.img_files.values())

    def iter(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx):
        dir = self.roots[0]
        img_timestamp_event_idx = self.img_timestamp_event_idx[dir]
        img = cv2.imread(self.img_files[dir][idx+1])
        img_timestamp, _, _ = img_timestamp_event_idx[idx+1]

        # events has form x, y, t_ns, p (in [0,1])
        t_events, xyp_events = self.event_data[dir]
        events = extract_events_from_memmap(t_events, xyp_events, idx, img_timestamp_event_idx)

        data = [
            {"events": events},
            {"frame": img, "timestamp": img_timestamp}
        ]

        return data

    


if __name__ == "__main__":
    import matplotlib.pyplot as plt 

    root = '/data/storage/michelle/ddd17_data'
    ds_indexable = DDD17SegmentationBase(root)

    for i in tqdm.tqdm(range(len(ds_indexable))):
        data_point = ds_indexable[i]
        print(len(data_point))
        print(data_point[0][0]['events0'].shape)
        print(data_point[1][0]['image0'].shape)

        fig, ax = plt.subplots(ncols=2, nrows=1)
        ax[0].imshow(data_point['events0'][:, :, 0])
        ax[0].set_title("events difference")
        ax[1].imshow(data_point['image0'][:, :, 0])
        ax[1].set_title("events difference")
        plt.show()

