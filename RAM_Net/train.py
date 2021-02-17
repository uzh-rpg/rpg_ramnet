import os
from os.path import join
import json
import logging
import argparse
import tqdm
import torch
from model.model import *
from model.loss import *
from model.metric import *
from torch.utils.data import DataLoader, ConcatDataset
from data_loader.dataset import *
from data_loader.dataset_mvsec import SequenceMVSEC, collate_mvsec
from trainer.lstm_trainer import LSTMTrainer
from utils.data_augmentation import Compose, RandomRotationFlip, RandomCrop, CenterCrop
from data_loader.dataset_asynchronous import my_collate
from os.path import join
import bisect

logging.basicConfig(level=logging.INFO, format='')


class ConcatDatasetCustom(ConcatDataset):
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx


def concatenate_subfolders(base_folder, base_folder2, dataset_type, dataset_type2, event_folder, depth_folder, frame_folder, sequence_length,
                           transform=None, proba_pause_when_running=0.0, proba_pause_when_paused=0.0, step_size=1, step_size2=1,
                           clip_distance=100.0, every_x_rgb_frame=1, normalize=True, scale_factor=1.0,
                           use_phased_arch=False, baseline=False, loss_composition=False, event_loader="voxelgrids",
                           nbr_of_events_per_voxelgrid=0, nbr_of_bins=5, dataset_idx_flag=False):

    """
    Create an instance of ConcatDataset by aggregating all the datasets in a given folder
    """

    subfolders = os.listdir(base_folder)
    print('Found {} samples in {}'.format(len(subfolders), base_folder))
    if base_folder2 is not None:
        subfolders2 = os.listdir(base_folder2)
        print('Dataset 2: Found {} samples in {}'.format(len(subfolders2), base_folder2))

    train_datasets = []
    for dataset_name in subfolders:
        train_datasets.append(eval(dataset_type)(base_folder=join(base_folder, dataset_name),
                                                 event_folder=event_folder,
                                                 depth_folder=depth_folder,
                                                 frame_folder=frame_folder,
                                                 sequence_length=sequence_length,
                                                 transform=transform,
                                                 proba_pause_when_running=proba_pause_when_running,
                                                 proba_pause_when_paused=proba_pause_when_paused,
                                                 step_size=step_size,
                                                 clip_distance=clip_distance,
                                                 every_x_rgb_frame=every_x_rgb_frame,
                                                 normalize=normalize,
                                                 scale_factor=scale_factor,
                                                 use_phased_arch=use_phased_arch,
                                                 baseline=baseline,
                                                 loss_composition=loss_composition,
                                                 event_loader=event_loader,
                                                 nbr_of_events_per_voxelgrid=nbr_of_events_per_voxelgrid,
                                                 nbr_of_bins=nbr_of_bins
                                                 ))
    if base_folder2 is not None:
        for dataset_name in subfolders2:
            train_datasets.append(eval(dataset_type2)(base_folder=join(base_folder2, dataset_name),
                                                     event_folder=event_folder,
                                                     depth_folder=depth_folder,
                                                     frame_folder=frame_folder,
                                                     sequence_length=sequence_length,
                                                     transform=transform,
                                                     proba_pause_when_running=proba_pause_when_running,
                                                     proba_pause_when_paused=proba_pause_when_paused,
                                                     step_size=step_size2,
                                                     clip_distance=clip_distance,
                                                     every_x_rgb_frame=every_x_rgb_frame,
                                                     normalize=normalize,
                                                     scale_factor=scale_factor,
                                                     use_phased_arch=use_phased_arch,
                                                     baseline=baseline,
                                                     loss_composition=loss_composition
                                                     ))

    if dataset_idx_flag == False:
        concat_dataset = ConcatDataset(train_datasets)
    elif dataset_idx_flag == True:
        concat_dataset = ConcatDatasetCustom(train_datasets)

    return concat_dataset


def main(config, resume, initial_checkpoint=None):
    train_logger = None

    L = config['trainer']['sequence_length']
    assert (L > 0)

    dataset_type, dataset_type2, base_folder, base_folder2, event_folder, depth_folder, frame_folder = {}, {}, {}, {}, {}, {}, {}
    proba_pause_when_running, proba_pause_when_paused = {}, {}
    step_size, step_size2 = {}, {}
    clip_distance = {}
    scale_factor = {}
    every_x_rgb_frame = {}
    baseline = {}
    event_loader = {}
    nbr_of_events_per_voxelgrid = {}
    nbr_of_bins = {}

    # this will raise an exception is the env variable is not set
    preprocessed_datasets_folder = os.environ['PREPROCESSED_DATASETS_FOLDER']

    use_phased_arch = config['use_phased_arch']

    for split in ['train', 'validation']:
        dataset_type[split] = config['data_loader'][split]['type']
        base_folder[split] = join(preprocessed_datasets_folder, config['data_loader'][split]['base_folder'])
        event_folder[split] = config['data_loader'][split]['event_folder']
        depth_folder[split] = config['data_loader'][split]['depth_folder']
        frame_folder[split] = config['data_loader'][split]['frame_folder']
        proba_pause_when_running[split] = config['data_loader'][split]['proba_pause_when_running']
        proba_pause_when_paused[split] = config['data_loader'][split]['proba_pause_when_paused']
        scale_factor[split] = config['data_loader'][split]['scale_factor']

        try:
            step_size[split] = config['data_loader'][split]['step_size']
        except KeyError:
            step_size[split] = 1

        try:
            step_size2[split] = config['data_loader'][split]['step_size2']
        except KeyError:
            step_size2[split] = 1

        try:
            clip_distance[split] = config['data_loader'][split]['clip_distance']
        except KeyError:
            clip_distance[split] = 100.0

        try:
            every_x_rgb_frame[split] = config['data_loader'][split]['every_x_rgb_frame']
        except KeyError:
            every_x_rgb_frame[split] = 1

        try:
            baseline[split] = config['data_loader'][split]['baseline']
        except KeyError:
            baseline[split] = False

        try:
            event_loader[split] = config['data_loader'][split]['event_loader']
        except KeyError:
            event_loader[split] = "voxelgrids"

        try:
            nbr_of_events_per_voxelgrid[split] = config['data_loader'][split]['nbr_of_events_per_voxelgrid']
        except KeyError:
            nbr_of_events_per_voxelgrid[split] = 0

        try:
            nbr_of_bins[split] = config['data_loader'][split]['nbr_of_bins']
        except KeyError:
            nbr_of_bins[split] = 5

        try:
            base_folder2[split] = join(preprocessed_datasets_folder, config['data_loader'][split]['base_folder2'])
        except KeyError:
            base_folder2[split] = None

        try:
            dataset_type2[split] = config['data_loader'][split]['type2']
        except KeyError:
            dataset_type2[split] = None


    loss_composition = config['trainer']['loss_composition']
    loss_weights = config['trainer']['loss_weights']
    normalize = config['data_loader'].get('normalize', True)

    train_dataset = concatenate_subfolders(base_folder['train'],
                                           base_folder2['train'],
                                           dataset_type['train'],
                                           dataset_type2['train'],
                                           event_folder['train'],
                                           depth_folder['train'],
                                           frame_folder['train'],
                                           sequence_length=L,
                                           transform=Compose([RandomRotationFlip(0.0, 0.5, 0.0),
                                                              RandomCrop(224)]),
                                           #transform=CenterCrop(112),
                                           proba_pause_when_running=proba_pause_when_running['train'],
                                           proba_pause_when_paused=proba_pause_when_paused['train'],
                                           step_size=step_size['train'],
                                           step_size2=step_size2['train'],
                                           clip_distance=clip_distance['train'],
                                           every_x_rgb_frame=every_x_rgb_frame['train'],
                                           normalize=normalize,
                                           scale_factor=scale_factor['train'],
                                           use_phased_arch=use_phased_arch,
                                           baseline=baseline['train'],
                                           loss_composition=loss_composition,
                                           event_loader=event_loader['train'],
                                           nbr_of_events_per_voxelgrid=nbr_of_events_per_voxelgrid['train'],
                                           nbr_of_bins=nbr_of_bins['train']
                                           )

    validation_dataset = concatenate_subfolders(base_folder['validation'],
                                                base_folder2['validation'],
                                                dataset_type['validation'],
                                                dataset_type2['validation'],
                                                event_folder['validation'],
                                                depth_folder['validation'],
                                                frame_folder['validation'],
                                                sequence_length=L,
                                                transform=CenterCrop(224),
                                                proba_pause_when_running=proba_pause_when_running['validation'],
                                                proba_pause_when_paused=proba_pause_when_paused['validation'],
                                                step_size=step_size['validation'],
                                                step_size2=step_size2['validation'],
                                                clip_distance=clip_distance['validation'],
                                                every_x_rgb_frame=every_x_rgb_frame['validation'],
                                                normalize=normalize,
                                                scale_factor=scale_factor['train'],
                                                use_phased_arch=use_phased_arch,
                                                baseline=baseline['validation'],
                                                loss_composition=loss_composition,
                                                event_loader=event_loader['validation'],
                                                nbr_of_events_per_voxelgrid=nbr_of_events_per_voxelgrid['validation'],
                                                nbr_of_bins=nbr_of_bins['validation']
                                                )

    # Set up data loaders
    kwargs = {'num_workers': config['data_loader']['num_workers'],
              'pin_memory': config['data_loader']['pin_memory']} if config['cuda'] else {}
    data_loader = DataLoader(train_dataset, batch_size=config['data_loader']['batch_size'],
                             shuffle=config['data_loader']['shuffle'], collate_fn=collate_mvsec, **kwargs)

    #for data in tqdm.tqdm(data_loader):
    #print("Testing dataloader...")
    #    pass

    #shuffle = config['data_loader']['shuffle'], collate_fn = my_collate, ** kwargs)

    valid_data_loader = DataLoader(validation_dataset, batch_size=config['data_loader']['batch_size'],
                                   shuffle=config['data_loader']['shuffle'], collate_fn=collate_mvsec, **kwargs)

    config['model']['gpu'] = config['gpu']
    config['model']['every_x_rgb_frame'] = config['data_loader']['train']['every_x_rgb_frame']
    config['model']['baseline'] = config['data_loader']['train']['baseline']
    config['model']['loss_composition'] = config['trainer']['loss_composition']

    torch.manual_seed(0)
    model = eval(config['arch'])(config['model'])

    if initial_checkpoint is not None:
        print('Loading initial model weights from: {}'.format(initial_checkpoint))
        checkpoint = torch.load(initial_checkpoint)
        # model.load_state_dict(checkpoint['state_dict'])
        if use_phased_arch:
            C, (H, W) = config["model"]["num_bins_events"], config["model"]["spatial_resolution"]
            dummy_input = torch.Tensor(1, C, H, W)
            times = torch.Tensor(1)
            _ = model.forward(dummy_input, times=times, prev_states=None)  # tag="events"
        model.load_state_dict(checkpoint['state_dict'])

    '''print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())'''

    model.summary()

    loss = eval(config['loss']['type'])
    loss_params = config['loss']['config'] if 'config' in config['loss'] else None
    print("Using %s with config %s" % (config['loss']['type'], config['loss']['config']))
    metrics = [eval(metric) for metric in config['metrics']]

    trainer = LSTMTrainer(model, loss, loss_params, metrics,
                          resume=resume,
                          config=config,
                          data_loader=data_loader,
                          valid_data_loader=valid_data_loader,
                          train_logger=train_logger)

    trainer.train()


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(
        description='Learning DVS Image Reconstruction')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-i', '--initial_checkpoint', default=None, type=str,
                        help='path to the checkpoint with which to initialize the model weights (default: None)')
    parser.add_argument('-g', '--gpu_id', default=None, type=int,
                        help='path to the checkpoint with which to initialize the model weights (default: None)')
    args = parser.parse_args()

    if args.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    config = None
    if args.resume is not None:
        if args.config is not None:
            logger.warning('Warning: --config overridden by --resume')
        if args.initial_checkpoint is not None:
            logger.warning(
                'Warning: --initial_checkpoint overriden by --resume')
        config = torch.load(args.resume)['config']
    if args.config is not None:
        config = json.load(open(args.config))
        path = os.path.join(config['trainer']['save_dir'], config['name'])
        if args.resume is None:
            assert not os.path.exists(path), "Path {} already exists!".format(path)
    assert config is not None

    main(config, args.resume, args.initial_checkpoint)
