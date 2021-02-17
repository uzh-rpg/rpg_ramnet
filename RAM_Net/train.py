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
from trainer.trainer_no_recurrent import TrainerNoRecurrent
from utils.data_augmentation import Compose, RandomRotationFlip, RandomCrop, CenterCrop
from os.path import join
import bisect

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

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


def concatenate_subfolders(base_folder, dataset_type, event_folder, depth_folder, frame_folder, sequence_length,
                           transform=None, proba_pause_when_running=0.0, proba_pause_when_paused=0.0, step_size=1,
                           clip_distance=100.0, every_x_rgb_frame=1, normalize=True, scale_factor=1.0,
                           use_phased_arch=False, baseline=False, loss_composition=False, reg_factor=5.7,
                           dataset_idx_flag=False, recurrency=True):
    """
    Create an instance of ConcatDataset by aggregating all the datasets in a given folder
    """

    subfolders = os.listdir(base_folder)
    print('Found {} samples in {}'.format(len(subfolders), base_folder))

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
                                                 reg_factor=reg_factor,
                                                 recurrency=recurrency))

    if dataset_idx_flag == False:
        concat_dataset = ConcatDataset(train_datasets)
    elif dataset_idx_flag == True:
        concat_dataset = ConcatDatasetCustom(train_datasets)

    return concat_dataset


def main(config, resume, initial_checkpoint=None):
    train_logger = None

    L = config['trainer']['sequence_length']
    assert (L > 0)

    dataset_type, base_folder, event_folder, depth_folder, frame_folder = {}, {}, {}, {}, {}
    proba_pause_when_running, proba_pause_when_paused = {}, {}
    step_size = {}
    clip_distance = {}
    scale_factor = {}
    every_x_rgb_frame = {}
    reg_factor = {}
    baseline = {}
    recurrency = {}

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

        if config['arch'] == "ERGB2Depth":
            recurrency[split] = False
        else:
            recurrency[split] = True

        try:
            step_size[split] = config['data_loader'][split]['step_size']
        except KeyError:
            step_size[split] = 1

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
            reg_factor[split] = config['data_loader'][split]['reg_factor']
        except KeyError:
            reg_factor[split] = False

    loss_composition = config['trainer']['loss_composition']
    loss_weights = config['trainer']['loss_weights']
    normalize = config['data_loader'].get('normalize', True)

    train_dataset = concatenate_subfolders(base_folder['train'],
                                           dataset_type['train'],
                                           event_folder['train'],
                                           depth_folder['train'],
                                           frame_folder['train'],
                                           sequence_length=L,
                                           transform=Compose([RandomRotationFlip(0.0, 0.5, 0.0),
                                                              RandomCrop(224)]),
                                           #transform=Compose([RandomRotationFlip(0.0, 0.5, 0.0),
                                           #                  CenterCrop([256, 512])]),
                                           #transform=CenterCrop(112),
                                           proba_pause_when_running=proba_pause_when_running['train'],
                                           proba_pause_when_paused=proba_pause_when_paused['train'],
                                           step_size=step_size['train'],
                                           clip_distance=clip_distance['train'],
                                           every_x_rgb_frame=every_x_rgb_frame['train'],
                                           normalize=normalize,
                                           scale_factor=scale_factor['train'],
                                           use_phased_arch=use_phased_arch,
                                           baseline=baseline['train'],
                                           loss_composition=loss_composition,
                                           reg_factor=reg_factor['train'],
                                           recurrency=recurrency['train']
                                           )

    validation_dataset = concatenate_subfolders(base_folder['validation'],
                                                dataset_type['validation'],
                                                event_folder['validation'],
                                                depth_folder['validation'],
                                                frame_folder['validation'],
                                                sequence_length=L,
                                                transform=CenterCrop(224),
                                                proba_pause_when_running=proba_pause_when_running['validation'],
                                                proba_pause_when_paused=proba_pause_when_paused['validation'],
                                                step_size=step_size['validation'],
                                                clip_distance=clip_distance['validation'],
                                                every_x_rgb_frame=every_x_rgb_frame['validation'],
                                                normalize=normalize,
                                                scale_factor=scale_factor['train'],
                                                use_phased_arch=use_phased_arch,
                                                baseline=baseline['validation'],
                                                loss_composition=loss_composition,
                                                reg_factor=reg_factor['validation'],
                                                recurrency=recurrency['validation']
                                                )

    # Set up data loaders
    kwargs = {'num_workers': config['data_loader']['num_workers'],
              'pin_memory': config['data_loader']['pin_memory']} if config['cuda'] else {}
    data_loader = DataLoader(train_dataset, batch_size=config['data_loader']['batch_size'],
                             shuffle=config['data_loader']['shuffle'], **kwargs)

    valid_data_loader = DataLoader(validation_dataset, batch_size=config['data_loader']['batch_size'],
                                   shuffle=config['data_loader']['shuffle'], **kwargs)

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

    '''if config['arch'] == "ERGB2Depth":
        trainer = TrainerNoRecurrent(model, loss, loss_params, metrics,
                                     resume=resume,
                                     config=config,
                                     data_loader=data_loader,
                                     valid_data_loader=valid_data_loader,
                                     train_logger=train_logger)'''
    #else:
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
