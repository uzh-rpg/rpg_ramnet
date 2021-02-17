import numpy as np
import torch
from base import BaseTrainer
from torchvision import utils
from model.loss import mse_loss, multi_scale_grad_loss
from utils.training_utils import select_evenly_spaced_elements, plot_grad_flow, plot_grad_flow_bars
import torch.nn.functional as f
from scipy.ndimage import zoom
import matplotlib.pyplot as plt


def quick_norm(img):
    return (img - torch.min(img)) / (torch.max(img) - torch.min(img) + 1e-5)


class LSTMTrainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
        self.optimizer is by default handled by BaseTrainer based on config.
    """

    def __init__(self, model, loss, loss_params, metrics, resume, config,
                 data_loader, valid_data_loader=None, train_logger=None):
        super(LSTMTrainer, self).__init__(model, loss,
                                          loss_params, metrics, resume, config, train_logger)
        self.config = config
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.valid = True if self.valid_data_loader is not None else False
        self.log_step = int(np.sqrt(self.batch_size))
        self.num_previews = config['trainer']['num_previews']
        self.num_val_previews = config['trainer']['num_val_previews']
        self.record_every_N_sample = 5
        self.movie = bool(config['trainer'].get('movie', True))
        self.still_previews = bool(config['trainer'].get('still_previews', False))
        self.grid_loss = bool(config['trainer'].get('grid_loss', False))
        self.every_x_rgb_frame = config['data_loader']['train']['every_x_rgb_frame']
        self.loss_composition = config['trainer']['loss_composition']
        self.loss_weights = config['trainer']['loss_weights']
        self.baseline = config['data_loader']['train']['baseline']
        self.calculate_total_metrics = []
        self.added_tensorboard_graph = False
        self.state_combination = config['model']['state_combination']

        self.state_preview_flag = False

        if config['use_phased_arch']:
            self.use_phased_arch = True
            print("Using phased architecture")
        else:
            self.use_phased_arch = False
            print("Will not use phased architecture")

        # Parameters for multi scale gradiant loss
        if 'grad_loss' in config:
            self.use_grad_loss = True
            try:
                self.weight_grad_loss = config['grad_loss']['weight']
            except KeyError:
                self.weight_grad_loss = 1.0

            print('Using Multi Scale Gradient loss with weight={:.2f}'.format(
                self.weight_grad_loss))
        else:
            print('Will not use Multi Scale Gradiant loss')
            self.use_grad_loss = False

        # Semantic loss (TO-DO)
        self.use_semantic_loss = False

        # Parameters for mse loss
        if 'mse_loss' in config:
            self.use_mse_loss = True
            try:
                self.weight_mse_loss = config['mse_loss']['weight']
            except KeyError:
                self.weight_mse_loss = 1.0

            try:
                self.mse_loss_downsampling_factor = config['mse_loss']['downsampling_factor']
            except KeyError:
                self.mse_loss_downsampling_factor = 0.5

            print('Using MSE loss with weight={:.2f} and downsampling factor={:.2f}'.format(
                self.weight_mse_loss, self.mse_loss_downsampling_factor))
        else:
            print('Will not use MSE loss')
            self.use_mse_loss = False

        # To visualize the progress of the network on the training and validation data,
        # we plot N training / validation samples, spread uniformly over all the samples
        self.preview_indices = select_evenly_spaced_elements(self.num_previews, len(self.data_loader))
        if valid_data_loader:
            self.val_preview_indices = select_evenly_spaced_elements(self.num_val_previews, len(self.valid_data_loader))

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        output = output.cpu().data.numpy()
        target = target.cpu().data.numpy()
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
        return acc_metrics

    def _to_input_and_target(self, item):
        events = item['events'].to(self.gpu)
        target = item['depth'].to(self.gpu)
        image = item['image'].to(self.gpu)
        semantic = item['semantic'].to(self.gpu) if self.use_semantic_loss else None
        times = item['times'].float().to(self.gpu) if self.use_phased_arch else None
        return events, image, target, flow, semantic, times

    @staticmethod
    def make_preview(event_previews, predicted_targets, groundtruth_targets):
        # event_previews: a list of [1 x 1 x H x W] event previews
        # predicted_frames: a list of [1 x 1 x H x W] predicted frames
        # for make_grid, we need to pass [N x 1 x H x W] where N is the number of images in the grid
        return utils.make_grid(torch.cat(event_previews + predicted_targets + groundtruth_targets, dim=0),
                               normalize=False, scale_each=True,
                               nrow=len(predicted_targets))

    @staticmethod
    def make_grad_loss_preview(grad_loss_frames):
        # grad_loss_frames is a list of N multi scale grad losses of size [1 x 1 x H x W]
        return utils.make_grid(torch.cat(grad_loss_frames, dim=0),
                               normalize=True, scale_each=True,
                               nrow=len(grad_loss_frames))

    @staticmethod
    def make_movie(event_previews, predicted_frames, groundtruth_targets):
        # event_previews: a list of [1 x 1 x H x W] event previews
        # predicted_frames: a list of [1 x 1 x H x W] predicted frames
        # for movie, we need to pass [1 x T x 1 x H x W] where T is the time dimension

        video_tensor = None
        for i in torch.arange(len(event_previews)):
            # voxel = quick_norm(event_previews[i])
            voxel = event_previews[i]
            predicted_frame = predicted_frames[i]  # quick_norm(predicted_frames[i])
            movie_frame = torch.cat([voxel,
                                     predicted_frame,
                                     groundtruth_targets[i]],
                                    dim=-1)
            movie_frame.unsqueeze_(dim=0)
            video_tensor = movie_frame if video_tensor is None else \
                torch.cat((video_tensor, movie_frame), dim=1)
        return video_tensor

    def calculate_losses(self, new_predicted_target, new_target, weight, loss_dict, record):
        # Compute the nominal loss
        if self.loss_params is not None:
            loss_dict['losses'].append(
                weight * self.loss(new_predicted_target, new_target, **self.loss_params))
        else:
            loss_dict['losses'].append(weight * self.loss(new_predicted_target, new_target))

        grad_loss_frames = []
        # Compute the multi scale gradient loss
        if self.use_grad_loss:
            if record:
                with torch.no_grad():
                    grad_loss_frames.append(multi_scale_grad_loss(new_predicted_target, new_target, preview=record))
            else:
                grad_loss = multi_scale_grad_loss(new_predicted_target, new_target)
                loss_dict['grad_losses'].append(weight * grad_loss)

        # Compute the mse loss
        if self.use_mse_loss:
            # compute the MSE loss at a lower resolution
            downsampling_factor = self.mse_loss_downsampling_factor

            if downsampling_factor != 1.0:
                new_target_downsampled = f.interpolate(
                    new_target, scale_factor=downsampling_factor, mode='bilinear', align_corners=False,
                    recompute_scale_factor=False)
                new_predicted_target_downsampled = f.interpolate(
                    new_predicted_target, scale_factor=downsampling_factor, mode='bilinear', align_corners=False,
                    recompute_scale_factor=False)
                mse = mse_loss(new_predicted_target_downsampled, new_target_downsampled)
            else:
                mse = mse_loss(new_predicted_target, new_target)
            loss_dict['mse_losses'].append(weight * mse)

        return loss_dict, grad_loss_frames

    def calculate_total_batch_loss(self, loss_dict, total_loss_dict, L):
        nominal_loss = sum(loss_dict['losses']) / float(L)
        #print("total si loss of batch: ", nominal_loss)

        losses = []
        losses.append(nominal_loss)

        # Add multi scale gradient loss
        if self.use_grad_loss:
            grad_loss = self.weight_grad_loss * sum(loss_dict['grad_losses']) / float(L)
            losses.append(grad_loss)
            #print("total grad loss of batch: ", grad_loss)

        # Add mse loss to the losses
        if self.use_mse_loss:
            mse = self.weight_mse_loss * sum(loss_dict['mse_losses']) / float(L)
            losses.append(mse)

        loss = sum(losses)

        # add all losses in a dict for logging
        with torch.no_grad():
            if not total_loss_dict:
                total_loss_dict = {'loss': loss, 'L_si': nominal_loss}
                if self.use_grad_loss:
                    total_loss_dict['L_grad'] = grad_loss
                if self.use_mse_loss:
                    total_loss_dict['L_mse'] = mse

            else:
                total_loss_dict['loss'] += loss
                total_loss_dict['L_si'] += nominal_loss
                if self.use_grad_loss:
                    total_loss_dict['L_grad'] += grad_loss
                if self.use_mse_loss:
                    total_loss_dict['L_mse'] += mse

        return total_loss_dict

    def forward_pass_sequence(self, sequence, record=False):
        # 'sequence' is a list containing L successive events <-> frames pairs
        # each element in 'sequence' is a dictionary containing the keys 'events' and 'frame'
        L = len(sequence)
        assert (L > 0)

        # list of per-iteration losses (summed after the loop)
        if record:
            predicted_targets = {}
            if self.state_preview_flag:
                state_previews = {}
            previews = {}
            grad_loss_frames = {}
            groundtruth_targets = []

        # initialize the K last predicted targets with -1
        N, _, H, W = sequence[0]['depth_image'].shape
        prev_states_lstm = {}
        for k in range(0, self.every_x_rgb_frame):
            prev_states_lstm['events{}'.format(k)] = None
            prev_states_lstm['depth{}'.format(k)] = None
        prev_states_lstm['image'] = None
        losses = {}
        total_batch_losses = {}
        new_target = None
        loss_dict = {'losses': [], 'grad_losses': [], 'mse_losses': []}

        prev_super_states = {'image': None}
        for l in range(L):
            item = sequence[l]

            '''fig, ax = plt.subplots(ncols=6, nrows=1)
            for ii in range(0,6):
                ax[ii].imshow(item["image"][0, ii, :, :])
                #ax[ii].imshow(element[l][key].cpu().numpy()[0])
            plt.show()'''
            
            # new_events, new_image, new_target, flow01, semantic, times = self._to_input_and_target(item)

            # the output of the network is a [N x 1 x H x W] tensor containing the image prediction
            # prev_super_states['image'] is given to model, because image network branch is always run as a last
            # step of one forward pass (events first, then image).
            new_predicted_targets, new_super_states, new_states_lstm = self.model(item,
                                                                                  prev_super_states['image'],
                                                                                  prev_states_lstm)

            grad_loss_frames_entries = {}
            for key, value in new_predicted_targets.items():
                if not self.loss_composition or key in self.loss_composition:
                    weight_idx = self.loss_composition.index(key)
                    # or (self.baseline == "e" and (l+1) % 5 == 0):
                    if key not in losses:
                        # losses[key] = {'losses': [], 'grad_losses': [], 'mse_losses': []}
                        losses[key] = loss_dict
                    new_target = item['depth_' + key].to(self.gpu)
                    is_nan = torch.isnan(new_target)
                    #print(new_predicted_targets[key][~is_nan].shape, torch.isnan(new_target).shape)
                    losses[key], grad_loss_frames_entries[key] = self.calculate_losses(new_predicted_targets[key],
                                                                                       new_target,
                                                                                       self.loss_weights[weight_idx],
                                                                                       losses[key], record)

            # calculate overall metrics (used for debugging):
            if not record:
                for key, prediction in new_predicted_targets.items():
                    metrics = self._eval_metrics(prediction, item['depth_' + key])
                    self.calculate_total_metrics.append(metrics)
            if record:
                with torch.no_grad():
                    if self.state_preview_flag:
                        if prev_super_states['image'] is not None and (l == 1 or l == int(L/2) or l == L-1):
                            if self.baseline == 'rgb':
                                loop_range = 1
                            elif self.baseline == 'e' or self.baseline == 'ergb0':
                                loop_range = self.every_x_rgb_frame
                            else:
                                loop_range = self.every_x_rgb_frame + 1
                            for ii in range(loop_range):
                                if ii == 0:
                                    # events0 or image for rgb baseline
                                    prev_state = prev_super_states['image']
                                    if self.baseline == 'rgb':
                                        current_state = new_super_states['image']
                                        key = 'image'
                                    else:
                                        current_state = new_super_states['events0']
                                        key = 'events0'
                                elif ii == loop_range-1:
                                    # last entry in batch = image. for e baselines, batch contains one less entry.
                                    prev_state = new_super_states['events{}'.format(loop_range - 2)]
                                    current_state = new_super_states['image']
                                    key = 'image'
                                else:
                                    # events1 - eventsX
                                    prev_state = new_super_states['events{}'.format(ii - 1)]
                                    current_state = new_super_states['events{}'.format(ii)]
                                    key = 'events{}'.format(ii)

                                state_preview_grid = None

                                for i in range(len(current_state)):
                                    if not bool(self.baseline) and self.state_combination == "convlstm":
                                        state_change = (current_state[i][0][0].cpu().numpy()  # .view(56, -1)
                                                        - prev_state[i][0][0].cpu().numpy())  # .view(56, -1)
                                    else:
                                        state_change = (current_state[i][0].cpu().numpy()  # .view(56, -1)
                                                        - prev_state[i][0].cpu().numpy())  # .view(56, -1)
                                    percentile_98 = np.percentile(np.abs(state_change), 98)
                                    eps = 1e-8
                                    state_change = state_change / (percentile_98 + eps)
                                    state_change = np.clip(state_change, -1, 1)
                                    dimension = state_change.shape[0]
                                    # only save 3 slices for preview out of the whole state tensor
                                    state_change = state_change[(0, int(dimension / 2), dimension - 1), :, :]
                                    state_change = zoom(state_change, (1, pow(2, i), pow(2, i)), order=1)
                                    concat_state_change = np.zeros([state_change.shape[1], state_change.shape[1] * 3])
                                    concat_state_change[:, 0:state_change.shape[1]] = state_change[0, :, :]
                                    concat_state_change[:,
                                    state_change.shape[1]:state_change.shape[1] * 2] = state_change[1, :, :]
                                    concat_state_change[:,
                                    state_change.shape[1] * 2:state_change.shape[1] * 3] = state_change[2, :, :]
                                    if state_preview_grid is None:
                                        state_preview_grid = concat_state_change
                                    else:
                                        state_preview_grid = np.append(state_preview_grid, concat_state_change, axis=0)
                                    # fig, ax = plt.subplots(ncols=1, nrows=1)
                                    # ax.imshow(state_preview_grid)
                                    # plt.show()

                                if key not in state_previews:
                                    state_previews[key] = []
                                if len(state_previews[key]) == 0:
                                    state_previews[key].append(state_preview_grid)
                                else:
                                    state_previews[key][-1] = np.append(state_previews[key][-1], state_preview_grid,
                                                                        axis=1)

                    for key, value in new_predicted_targets.items():
                        if key not in predicted_targets:
                            predicted_targets[key] = []
                        predicted_targets[key].append(new_predicted_targets[key].clone())
                        if key not in previews:
                            previews[key] = []
                        previews[key].append(torch.sum(item[key].to(self.gpu), dim=1).unsqueeze(0))
                        if not self.loss_composition or key in self.loss_composition:
                            #(self.baseline == "e" and l+1 % 5 == 0):
                            if key not in grad_loss_frames:
                                grad_loss_frames[key] = []
                            grad_loss_frames[key].extend(grad_loss_frames_entries[key])
                    groundtruth_targets.append(new_target.clone())

            prev_states_lstm = new_states_lstm
            prev_super_states = new_super_states
        for key, value in losses.items():
            total_batch_losses = self.calculate_total_batch_loss(losses[key], total_batch_losses, L)
            #print("total batch loss: ", key, total_batch_losses['loss'])

        return total_batch_losses, \
            predicted_targets if record else None, \
            groundtruth_targets if record else None, \
            previews if record else None, \
            grad_loss_frames if record else None, \
            state_previews if (record and self.state_preview_flag) else None

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """

        # uncomment to see visualization of input data
        '''image_old = None
        for index, element in enumerate(self.data_loader.dataset):
            print('index', index)
            for l in range(3):
                #image_new1 = element[l]['events0'].cpu().numpy()[-1]
                #image_new2 = element[l]['image'].cpu().numpy()[-1]
                #print(image_new1.shape)
                #if image_old is not None:
                #    fig, ax = plt.subplots(ncols=3, nrows=1)
                #    ax[0].imshow(image_new1)
                #    ax[1].imshow(image_old)
                #    ax[2].imshow(image_new1-image_old)
                fig, ax = plt.subplots(ncols=6, nrows=3)
                index_1 = 0
                index_2 = 0
                for i, key in enumerate(element[l].keys()):
                    if "depth" in key:
                        ax[0, index_1].imshow(element[l][key].cpu().numpy()[0])
                        ax[0, index_1].set_title("groundtruth " + key)
                        index_1 += 1
                    else:
                        # ax[2, index_2].imshow(torch.sum(input[key], dim=1)[0].cpu().numpy())  # all
                        ax[1, index_2].imshow(torch.sum(element[l][key][0:-2], dim=0).cpu().numpy())  # events only
                        ax[2, index_2].imshow(element[l][key][-1].cpu().numpy())  # image only
                        ax[1, index_2].set_title("input eventdata" + key)
                        ax[2, index_2].set_title("input imagedata" + key)
                        index_2 += 1
                plt.show()

                #image_old = image_new2
                #sequence = self.data_loader.dataset[index]'''

        self.model.train()

        all_losses_in_batch = {}
        for batch_idx, sequence in enumerate(self.data_loader):
            self.optimizer.zero_grad()
            losses, _, _, _, _, _ = self.forward_pass_sequence(sequence)
            loss = losses['loss']
            # loss_images.backward(retain_graph=True)
            loss.backward()
            if batch_idx % 25 == 0:
                plot_grad_flow(self.model.named_parameters())
            self.optimizer.step()

            '''for tag, value in self.model.named_parameters():
                print("tag: ", tag)
                print("value data: ", value.data.type())
                print("value grad: ", value.grad.type())'''
            # print("Model's state_dict:")
            # for param_tensor in self.model.state_dict():
            #    print(param_tensor, "\t", self.model.state_dict()[param_tensor].requires_grad)

            with torch.no_grad():
                for loss_name, loss_value in losses.items():
                    if loss_name not in all_losses_in_batch:
                        all_losses_in_batch[loss_name] = []
                    all_losses_in_batch[loss_name].append(loss_value.item())

                if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                    loss_str = ''
                    for loss_name, loss_value in losses.items():
                        loss_str += '{}: {:.4f} '.format(loss_name, loss_value.item())
                    self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] {}'.format(
                        epoch,
                        batch_idx * self.data_loader.batch_size,
                        len(self.data_loader) * self.data_loader.batch_size,
                        100.0 * batch_idx / len(self.data_loader),
                        loss_str))

        with torch.no_grad():
            # create a set of previews and log them
            previews = []
            if self.state_preview_flag:
                previews_states = {}
            total_metrics = np.zeros(len(self.metrics))
            self.preview_count = 0

            for preview_idx in self.preview_indices:
                # data is a sequence containing L successive events <-> frames pairs
                sequence = self.data_loader.dataset[preview_idx]

                # every element in sequence is a [C x H x W] tensor
                # but the model requires [1 x C x H x W] tensor, so
                # we preprocess the data here to adjust to this expected format
                '''for i in range(len(sequence)):
                    print("lstm trainer:", len(sequence), sequence[i]['depth_image'].shape)'''
                for data_items in sequence:
                    for key, item in data_items.items():
                        if key != "depth_image" or len(item.shape) < 4:
                            item.unsqueeze_(dim=0)

                _, predicted_targets, groundtruth_targets, previews_outputs, grad_loss_frames, \
                    state_previews = self.forward_pass_sequence(sequence, record=True)

                fig = plot_grad_flow_bars(self.model.named_parameters())
                self.writer.add_figure('grad_figure', fig, global_step=epoch)
                for key in predicted_targets.keys():
                    hist_idx = len(predicted_targets[key]) - 1  # choose an idx to plot
                    self.writer.add_histogram(f'{self.preview_count}_prediction_{key}',
                                              predicted_targets[key][hist_idx],
                                              global_step=epoch)
                    self.writer.add_histogram(f'{self.preview_count}_groundtruth_{key}',
                                              groundtruth_targets[hist_idx],
                                              global_step=epoch)
                    total_metrics += self._eval_metrics(predicted_targets[key][0], groundtruth_targets[0])


                    if self.movie:
                        video_tensor = self.make_movie(previews_outputs[key], predicted_targets[key],
                                                              groundtruth_targets)
                        self.writer.add_video(
                            f'movie_{self.preview_count}__{key}__prediction__groundtruth',
                            video_tensor, global_step=epoch, fps=5)
                    if self.still_previews:
                        step = self.record_every_N_sample
                        if self.state_preview_flag:
                            if key not in previews_states:
                                previews_states[key] = []
                            for i in range(len(state_previews[key])):
                                previews_states[key].append(state_previews[key][i][None, :])
                        previews.append(self.make_preview(
                            previews_outputs[key][::step], predicted_targets[key][::step], groundtruth_targets[::step]))

                    if self.grid_loss and (not self.loss_composition or key in self.loss_composition):
                        # or (self.baseline == "e" and l+1 % 5 == 0)):
                        if len(grad_loss_frames[key]) != 0:
                            # print ("train len(grad_loss_frames[0]): ", len(grad_loss_frames[::step][0]))
                            previews.append(self.make_grad_loss_preview(grad_loss_frames[key][::step][0]))

                for tag, value in self.model.named_parameters():
                    # print("tag: ", tag)
                    # print("value data: ", value.data)
                    # print("value grad max: ", torch.max(value.grad), "min: ", torch.min(torch.abs(value.grad)))
                    if value.grad is not None:
                        self.writer.add_histogram(tag + '/grad', value.grad, global_step=epoch)
                    else:
                        print("Skipped adding gradient histogram of ", tag, ", because the gradient is None.")
                    self.writer.add_histogram(tag + '/weights', value.data, global_step=epoch)

                self.preview_count += 1

        # compute average losses over the batch
        total_losses = {loss_name: sum(loss_values) / len(self.data_loader)
                        for loss_name, loss_values in all_losses_in_batch.items()}
        log = {
            'loss': total_losses['loss'],
            'losses': total_losses,
            'metrics': (total_metrics / self.num_previews).tolist(),
            'previews': previews
        }
        if self.state_preview_flag:
            log['previews_states'] = previews_states
        #print("num_previes: ", self.num_previews)
        #print("log metrics: ", log['metrics'])

        #print("total metrics: ", np.sum(np.array(self.calculate_total_metrics), 0) / len(self.calculate_total_metrics))

        if self.valid:
            val_log = self._valid_epoch(epoch=epoch)
            log = {**log, **val_log}

        return log

    def _valid_epoch(self, epoch=0):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        all_losses_in_batch = {}
        with torch.no_grad():
            for batch_idx, sequence in enumerate(self.valid_data_loader):
                losses, _, _, _, _, _ = self.forward_pass_sequence(sequence)
                for loss_name, loss_value in losses.items():
                    if loss_name not in all_losses_in_batch:
                        all_losses_in_batch[loss_name] = []
                    all_losses_in_batch[loss_name].append(loss_value.item())

                if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                    self.logger.info('Validation: [{}/{} ({:.0f}%)]'.format(
                        batch_idx * self.valid_data_loader.batch_size,
                        len(self.valid_data_loader) * self.valid_data_loader.batch_size,
                        100.0 * batch_idx / len(self.valid_data_loader)))
            print("all losses in batch in validation: ", all_losses_in_batch)

            # create a set of previews and log then
            val_previews = []
            total_metrics = np.zeros(len(self.metrics))
            self.preview_count = 0
            for val_preview_idx in self.val_preview_indices:
                # data is a sequence containing L successive events <-> frames pairs
                sequence = self.valid_data_loader.dataset[val_preview_idx]

                # every element in sequence is a [C x H x W] tensor
                # but the model requires [1 x C x H x W] tensor, so
                # we preprocess the data here to adjust to this expected format
                for data_items in sequence:
                    for key, item in data_items.items():
                        if key != "depth_image" or len(item.shape) < 4:
                            item.unsqueeze_(dim=0)

                _, predicted_targets, groundtruth_targets, previews_outputs, \
                    grad_loss_frames, states_previews \
                    = self.forward_pass_sequence(sequence, record=True)

                for key in predicted_targets.keys():
                    total_metrics += self._eval_metrics(predicted_targets[key][0], groundtruth_targets[0])
                    if self.movie:
                        video_tensor = self.make_movie(previews_outputs[key], predicted_targets[key], groundtruth_targets)
                        self.writer.add_video(
                            f"val_movie_{self.preview_count}__{key}__prediction__groundtruth",
                            video_tensor, global_step=epoch, fps=5)
                        self.preview_count += 1
                    if self.still_previews:
                        step = self.record_every_N_sample
                        val_previews.append(self.make_preview(
                            previews_outputs[key][::step], predicted_targets[key][::step], groundtruth_targets[::step]))
                    if self.grid_loss and (not self.loss_composition or key in self.loss_composition):
                        # or (self.baseline == "e" and l+1 % 5 == 0)):
                        if len(grad_loss_frames[key]) != 0:
                            val_previews.append(self.make_grad_loss_preview(grad_loss_frames[key][::step][0]))

        total_losses = {loss_name: sum(loss_values) / len(self.valid_data_loader)
                        for loss_name, loss_values in all_losses_in_batch.items()}
        return {
            'val_loss': total_losses['loss'],
            'val_losses': total_losses,
            'val_metrics': (total_metrics / self.num_val_previews).tolist(),
            'val_previews': val_previews
        }
