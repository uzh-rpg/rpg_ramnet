# Config:
In the following you can find explanations of different parameters in the config whose meaning is not apparent by the name.

### name: 
The checkpoint will be saved under this name in the specified save directory (trainer, save_dir). Make sure that there does not exist a checkpoint at this location with the same name yet, otherwise it will not work.

## dataloader:
### type:
Specifies the type of dataloader, either SequenceMVSEC or SequenceSynchronizedFramesEventsDataset can be used.
### type2:
For the branch `asynchronous_irregular_real_data` a second dataset can be specified in order to train on two datasets simultaneously. Also specify the base_folder2 and step_size2 if this is used.
### base_folder:
specify the path of the data folder, starting from the exported path (see README in AMM-Net folder)
### step_size:
specifies the number of skipped datapoints before beginning a new sequence. If sequence = 5 and step_size = 5, each datapoint is only seen one during an epoch. If the the step_size is smaller than the sequence_length, datapoints are seen several times. However, step_size > 0 does not actually skip image frames that the network sees, it only defines where a new sequence should be start with respect to the starting data point from the last sequence.
### clip_distance:
Defines the max seen depth used to calculate metric depth from log depth. When training a network for further use in MVSEC, the same clip_distance as in MVSEC should be used for the training in simulation (=3.70378).
### every_x_rgb_frames:
defines how many rgb frames are skipped in order to get asynchronous data read in. should be equal to 1 if baseline = "ergb0" / "e", otherwise events are skipped.

### scale_factor:
downscales inputs for faster training. 
### baseline:
If `false`, AMM-Net is trained. Other options are `rgb`, `e`, `ergb0`. For the `asynchronous_irregular_real_data` branch only `false` and `rgb` can be used.


## trainer:
### loss_combination 
Defines which predictions are used to calculate the loss. It can be false (=use loss of all inputs of the network, e.g. events0, events1, events2, ..., image), or can be a list of the inputs that should be used, e.g. simply "image" or ["image", "events5"].
For the branch `asynchronous_irregular_real_data`, ["image_last", "events_last"] should be used as depth data is not available at intermediate timesteps.
For the baselines in simulation, it should be equal to "image".
## model:
### state_combination:
"convgru" and "convlstm" can be used. Defines how the state update is performed.
For the baselines, it should be defined as "convlstm", this will be the recurrent layer in the encoder.
### spatial_resolution:
Only needed for the use with phased LSTM to properly load the model at test time. However, as phased LSTM is not used in the tested configurations, this could be omitted.



