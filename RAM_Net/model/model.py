from base import BaseModel
from itertools import compress
import copy
import numpy as np
import torch.nn as nn
import torch
import time
from model.statenet import StateNetPhasedRecurrent
from os.path import join
from model.submodules import \
    ConvLSTM, ResidualBlock, ConvLayer, \
    UpsampleConvLayer, TransposedConvLayer

class BaseERGB2Depth(BaseModel):
    def __init__(self, config):
        super().__init__(config)

        assert ('num_bins_rgb' in config)
        self.num_bins_rgb = int(config['num_bins_rgb'])  # number of bins in the rgb tensor
        assert ('num_bins_events' in config)
        self.num_bins_events = int(config['num_bins_events'])  # number of bins in the voxel grid event tensor

        try:
            self.skip_type = str(config['skip_type'])
        except KeyError:
            self.skip_type = 'sum'

        try:
            self.state_combination = str(config['state_combination'])
        except KeyError:
            self.state_combination = 'sum'

        try:
            self.num_encoders = int(config['num_encoders'])
        except KeyError:
            self.num_encoders = 4

        try:
            self.base_num_channels = int(config['base_num_channels'])
        except KeyError:
            self.base_num_channels = 32

        try:
            self.num_residual_blocks = int(config['num_residual_blocks'])
        except KeyError:
            self.num_residual_blocks = 2

        try:
            self.recurrent_block_type = str(config['recurrent_block_type'])
        except KeyError:
            self.recurrent_block_type = 'convlstm'

        try:
            self.norm = str(config['norm'])
        except KeyError:
            self.norm = None

        try:
            self.use_upsample_conv = bool(config['use_upsample_conv'])
        except KeyError:
            self.use_upsample_conv = True

        try:
            self.every_x_rgb_frame = config['every_x_rgb_frame']
        except KeyError:
            self.every_x_rgb_frame = 1

        try:
            self.baseline = config['baseline']
        except KeyError:
            self.baseline = False

        try:
            self.loss_composition = config['loss_composition']
        except KeyError:
            self.loss_composition = False

        self.kernel_size = int(config.get('kernel_size', 5))
        self.gpu = torch.device('cuda:' + str(config['gpu']))


class ERGB2DepthRecurrent(BaseERGB2Depth):
    """
    Recurrent, UNet-like architecture where each encoder is followed by a ConvLSTM or ConvGRU.
    """

    def __init__(self, config):
        super(ERGB2DepthRecurrent, self).__init__(config)

        '''print(self.num_bins_rgb, self.num_bins_events, 1, self.skip_type, self.state_combination, 'sigmoid',
              self.num_encoders, self.base_num_channels, self.num_residual_blocks, self.norm, self.use_upsample_conv,
              self.recurrent_block_type, self.baseline)'''

        self.statenetphasedrecurrent = StateNetPhasedRecurrent(num_input_channels_rgb=self.num_bins_rgb,
                                                               num_input_channels_events=self.num_bins_events,
                                                               num_output_channels=1,
                                                               skip_type=self.skip_type,
                                                               state_combination=self.state_combination,
                                                               activation='sigmoid',
                                                               num_encoders=self.num_encoders,
                                                               base_num_channels=self.base_num_channels,
                                                               num_residual_blocks=self.num_residual_blocks,
                                                               norm=self.norm,
                                                               use_upsample_conv=self.use_upsample_conv,
                                                               recurrent_block_type=self.recurrent_block_type,
                                                               baseline=self.baseline)
        self.max_num_channels = self.base_num_channels * pow(2, self.num_encoders)
        self.time_measurements_events = []
        self.time_measurements_images = []

    def forward(self, item, prev_super_states_batch, prev_states_lstm):
        # create lists for return
        predictions_dict_batch, super_states_batch, states_lstm_dict_batch = [], [], []


        # loop over whole batch
        for i, batch_entry in enumerate(item):
            #print("keys: ", [list(items.keys())[0] for items in batch_entry])
            if len(batch_entry) > 2:

                predictions_dict, states_lstm_dict = {}, {}
                # if not self.baseline:
                last_lstm_state_events = prev_states_lstm[i]["events_last"]
                last_lstm_state_image = prev_states_lstm[i]["image_last"]

                # get indexes of last events and image
                last_event_index = None
                if not self.baseline:
                    for k, entry in reversed(list(enumerate(batch_entry))):
                        if "events" in list(entry.keys())[0]:
                            last_event_index = k
                            break

                last_image_index = None
                for k, entry in reversed(list(enumerate(batch_entry))):
                    if "image" in list(entry.keys())[0]:
                        last_image_index = k
                        break
                #print("last indexes: ", last_event_index, last_image_index)

                prev_super_states = prev_super_states_batch[i]

                # initialize state if state is not defined yet
                if prev_super_states is None:
                    # print("state is none, state is initialized")
                    prev_super_states = []
                    B = 1  # state is only defined separately for each entry of whole batch, therefore B=1
                    _, H, W = list(item[0][0].values())[0].shape
                    for i in range(self.num_encoders):
                        H_state = int(H / pow(2, i + 1))
                        W_state = int(W / pow(2, i + 1))
                        N_channels = int(self.base_num_channels * pow(2, i + 1))
                        if not bool(self.baseline) and self.state_combination == 'convlstm':
                            # for state with convlstm, state has two entries (hidden & cell state)
                            prev_super_states.append([torch.zeros([B, N_channels, H_state, W_state]).to(self.gpu),
                                                      torch.zeros([B, N_channels, H_state, W_state]).to(self.gpu)])
                        else:
                            prev_super_states.append(torch.zeros([B, N_channels, H_state, W_state]).to(self.gpu))

                # loop over all image and events datasets
                for j, data_item in enumerate(batch_entry):
                    key = list(data_item.keys())[0]
                    if "depth" not in key and key != "img_loss":
                        input_tensor = data_item[key].to(self.gpu)

                        #t0 = time.time()
                        if "events" in key:
                            # run through event encoder and decoder
                            super_states, states_lstm_events = \
                                self.statenetphasedrecurrent.forward_events(input_tensor[None, :], prev_super_states,
                                                                            last_lstm_state_events, None)
                            prediction = self.statenetphasedrecurrent.forward_decoder(super_states)
                            last_lstm_state_events = states_lstm_events
                            #t1 = time.time() - t0
                            #self.time_measurements_events.append(t1)
                            #print("Timing statistics for events: ", np.mean(np.asarray(self.time_measurements_events)),
                            #      min(self.time_measurements_events), max(self.time_measurements_events))

                        elif "image" in key:
                            # run through image encoder and decoder
                            super_states, states_lstm_image = \
                                self.statenetphasedrecurrent.forward_images(input_tensor[None, :], prev_super_states,
                                                                            last_lstm_state_image, None)
                            prediction = self.statenetphasedrecurrent.forward_decoder(super_states)
                            #t1 = time.time() - t0
                            #self.time_measurements_images.append(t1)
                            last_lstm_state_image = states_lstm_image
                            #print("Timing statistics for images: ", np.mean(np.asarray(self.time_measurements_images)),
                            #    min(self.time_measurements_images), max(self.time_measurements_images))

                        predictions_dict[key] = prediction
                        prev_super_states = super_states

                        if j == last_event_index or j == last_image_index:
                            if j == last_event_index:
                                states_lstm_dict["events_last"] = last_lstm_state_events
                                predictions_dict["events_last"] = prediction
                            else:
                                states_lstm_dict["image_last"] = last_lstm_state_image
                                predictions_dict["image_last"] = prediction

                        if last_image_index is None:
                            states_lstm_dict["image_last"] = last_lstm_state_image
                        if last_event_index is None:
                            states_lstm_dict["events_last"] = last_lstm_state_events


                predictions_dict_batch += [predictions_dict]
                super_states_batch += [super_states]
                states_lstm_dict_batch += [states_lstm_dict]
            else:
                predictions_dict_batch += [{}]
                super_states_batch += [prev_super_states_batch[i]]
                states_lstm_dict_batch += [{"image_last": prev_states_lstm[i]["image_last"],
                                           "events_last": prev_states_lstm[i]["events_last"]}]
        return predictions_dict_batch, super_states_batch, states_lstm_dict_batch
