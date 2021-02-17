## Basics 
To conduct the following experiments, one has to export the path to the folder with all datasets:

```bash
export PREPROCESSED_DATASETS_FOLDER=/data/storage/michelle
```

If experiments on simulation data are conducted with the dataloader 2 (regular-asynchronous data), the master branch can be used.
For experiments on irregular-asynchronous data like MVSEC (dataloader 3), the `asynchronous_irregular_real_data` branch can be used. Pay attention that this branch is slower in training than the others as a batch cannot be fed as a whole into the pytorch layers because of the varying data structure of the different datapackages. It therefore has to loop through the batch during the forward pass which results in slower training.


## Training:
For training, the config has to be defined that contains all training parameters. A more detailed explanation of the config parameters can be found in the README in the /config folder.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/train_e2depth_si_grad_loss_statenet_ergb.json `
```

If a training should be started with an initial checkpoint it can be added as follows:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/train_e2depth_si_grad_loss_statenet_ergb.json --initial_checkpoint /data/storage/michelle/e2depth_checkpoints/EventScape/e2depth_si_grad_loss_statenet_skip_conv_convgru_every5rgbframes_imageevents4loss_LR0003_S5_100_scale1/model_best.pth.tar`
```
To resume a training, use:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --resume /data/storage/michelle/e2depth_checkpoints/EventScape/e2depth_si_grad_loss_statenet_skip_conv_convgru_every5rgbframes_imageevents4loss_LR0003_S5_100_scale1/checkpoint-epoch100-loss-0.0682.pth.tar
```


## Testing:
### Test script
Testing a network is done by using two scripts. First, the `test.py` script is used to save the outputs of the network.
As a second step, the `evaluation.py` script is used to calculate the metrics based on these outputs.

For the test script, the path to the model that needs to be tested needs to be specified, as well as the output folder for saving the data (if no output folder is specified the data is not saved) and a data folder, containing the test dataset:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py --path_to_model /data/storage/michelle/e2depth_checkpoints/EventScape/e2depth_si_grad_loss_statenet_skip_convlstm_baseline_ergb_every5rgbframes_LR0003_S5_100_scale1/model_best.pth.tar --output_path /data/storage/michelle/e2depth_evaluation/test --data_folder /data/storage/michelle/dataset_mathias_23_07/test/
```
The test script outputs npy arrays for the following evaluation script as well as different png images for visual results.

### Evaluation script
The evaluation script takes a target a predictions dataset as inputs. In addition, an outputfolder can be specified if plots of the results should be safed. By adding `--debug --idx 10`, plots are shown for the index 10.
The target and test dataset path define, on which predictions the metrics are calculated. For example, if the target_dataset is specified as `.../ground_truth/npy/depth_image/` and the predictions_dataset as `.../npy/image/`, the image predictions are analyzed. For `.../ground_truth/npy/depth_events4/` and  `.../npy/events4/` the metrics after the 5th event voxelgrid are analyzed.
For the MVSEC dataset, only one depth label is available per datapackage. Therefore the `.../ground_truth/npy` folder only contains a `/depth` subfolder. The predictions that are time aligned with this are in the `.../npy/events_last/` folder for RAM-Net and in the `.../npy/image_last/` folder for the RGB baseline (=I baseline).
For the `evaluation.py` script, the clip distance and reg_factor need to be specified. They are used to calculate the metric depth out of the log depth maps.

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation.py --target_dataset /data/storage/michelle/e2depth_evaluation/EventScape_final/baseline_rgb_clip1000/ground_truth/npy/depth_image/ --predictions_dataset /data/storage/michelle/e2depth_evaluation/EventScape_final/baseline_rgb_clip1000/npy/image/ --clip_distance 1000 --reg_factor 5.70378
```



## Training of AMM-Net
The final training checkpoints of RAM-Net can be found in `/data/storage/michelle/e2depth_checkpoints` under `EventScape` and `mvsec_training`
The outputs from the `test.py` script for the final testing can be found in `/data/storage/michelle/e2depth_evaluation` in `EventScape_final` and `mvsec_final`
