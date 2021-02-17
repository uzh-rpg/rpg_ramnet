from base import BaseModel
import torch.nn as nn
import torch
from model.unet import UNet
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

class ERGB2Depth(BaseERGB2Depth):
    def __init__(self, config):
        super(ERGB2Depth, self).__init__(config)

        self.unet = UNet(num_input_channels=self.num_bins_rgb,
                         num_output_channels=1,
                         skip_type=self.skip_type,
                         activation='sigmoid',
                         num_encoders=self.num_encoders,
                         base_num_channels=self.base_num_channels,
                         num_residual_blocks=self.num_residual_blocks,
                         norm=self.norm,
                         use_upsample_conv=self.use_upsample_conv)

    def forward(self, item, prev_super_states, prev_states_lstm):
        #def forward(self, event_tensor, prev_states=None):
        """
        :param event_tensor: N x num_bins x H x W
        :return: a predicted image of size N x 1 x H x W, taking values in [0,1].
        """
        predictions_dict = {}
        '''for key in item.keys():
            if "depth" not in key:
                event_tensor = item[key].to(self.gpu)

                prediction = self.unet.forward(event_tensor)
                predictions_dict[key] = prediction'''

        event_tensor = item["image"].to(self.gpu)
        prediction = self.unet.forward(event_tensor)
        predictions_dict["image"] = prediction

        return predictions_dict, {'image': None}, prev_states_lstm


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

    def forward(self, item, prev_super_states, prev_states_lstm):

        predictions_dict, super_state_dict, states_lstm_dict = {}, {}, {}

        # initialize state if state is not defined yet
        if prev_super_states is None:
            # print("state is none, state is initialized")
            prev_super_states = []
            B, C, H, W = item['image'].shape
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

        if not bool(self.baseline) or self.baseline == "ergb0" \
                or (self.baseline == "e" and self.loss_composition == "image"):
            # events are only processed for these combinations.

            if self.baseline == "ergb0" or (self.baseline == "e" and self.loss_composition == "image"):
                loop_range = self.every_x_rgb_frame - 1
                last_lstm_state = prev_states_lstm['image']
                # for baselines, the same encoders is used for image or event input, so the last_lstm_state is
                # the one from the last image.
            else:
                loop_range = self.every_x_rgb_frame
                last_lstm_state = prev_states_lstm['events{}'.format(self.every_x_rgb_frame - 1)]
                # for statenet, events and images have different encoders, so the last_lstm_state is the one
                # from the last event of the previous sequence.

            for k in range(loop_range):
                event_tensor = item['events{}'.format(k)].to(self.gpu)
                times = None
                # implement if phased architecture is used!
                #times = item['times'].float().to(self.gpu) if self.use_phased_arch else None
                if self.baseline == "ergb0" or self.baseline == 'e':
                    # baselines don't have an event encoder, so forward_images is used here.
                    super_states_events, states_lstm_events = \
                        self.statenetphasedrecurrent.forward_images(event_tensor, prev_super_states,
                                                                    last_lstm_state, times)
                else:
                    super_states_events, states_lstm_events = \
                        self.statenetphasedrecurrent.forward_events(event_tensor, prev_super_states,
                                                                    last_lstm_state, times)

                prediction_events = self.statenetphasedrecurrent.forward_decoder(super_states_events)

                predictions_dict['events{}'.format(k)] = prediction_events
                super_state_dict['events{}'.format(k)] = super_states_events
                states_lstm_dict['events{}'.format(k)] = states_lstm_events
                prev_super_states = super_states_events
                last_lstm_state = states_lstm_events
                # reset lstm_state for next loop round.

        image_tensor = item['image'].to(self.gpu)
        times = None

        if not bool(self.baseline) or self.baseline == "rgb" \
                or (self.baseline == "e" and self.loss_composition != "image"):
            # for statenet, there is a seperate image encoder which needs the last_lstm_state to be the one last
            # seen by this encoder. For the baselines that don't have events inbetween frames, the last seen lstm
            # state is also the "image" one.
            last_lstm_state = prev_states_lstm['image']

        super_states_images, states_lstm_images = \
            self.statenetphasedrecurrent.forward_images(image_tensor, prev_super_states,
                                                        last_lstm_state, times)
        prediction_images = self.statenetphasedrecurrent.forward_decoder(super_states_images)

        predictions_dict['image'] = prediction_images
        super_state_dict['image'] = super_states_images
        states_lstm_dict['image'] = states_lstm_images

        return predictions_dict, super_state_dict, states_lstm_dict
