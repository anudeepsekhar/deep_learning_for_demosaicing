# Demosaicing with Deep Learning

Video summary: https://www.youtube.com/watch?v=PaszhVmOJBU

Report: https://drive.google.com/file/d/1iTXtseetwVAs5n_7BtZtqHeuIGCPeNtz/view?usp=sharing

## Dataset
To obtain the CUB200 dataset, follow these steps:
1. Create `data` folder if it does not exist yet
2. Manually download the `.tgz` file from this link (http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz) to your local machine
3. Upload the `.tgz` to your `data` folder
4. Run `tar -xvzf "./data/CUB200/CUB_200_2011.tgz"`
5. Manually move the extracted `CUB_200_2011` to the `data` folder
6. Manually delete the `.tgz` file from the `data` folder

## Joint demosaicing and superresolution
### Data preprocessing
Run the command `preprocess.py --crop <"true" or "false">`

Argument options:

* `crop` (optional): `true` or `false`. Defaults to `false`. Create input Bayer and ground truth images with cropping. 

Note:
* The uncropped images created will be saved to `CUB200_processed` in `data`
* The cropped images created will be saved to `CUB200_processed_not_cropped` in `data`

### Training
Run the command `train_main.py --resume_from_ckp <true or false> --trial_number <an integer> --model <model name> --num_epochs <an integer> --lr <a float number>`

Argument options:

* `resume_from_ckp`: `true`, `false`

* `trial_number`: the trial number

* `model`: `deep_residual_network`,`deep_residual_network_rednet`,`deep_residual_network_SRCNN`,`rednet`,`SRCNN`,`VDSR`

* `num_epochs`: number of epochs

* `lr` (optional): between `1e-6` and `1e-1`, defaults to `1e-4` if not given

* `not_cropped` (optional): either `true` or `false`. Use uncropped Input Bayer and ground truth images. Defaults to `false`. This argument only has an effect when `model` is `SRCNN`.

### Testing
Run the command `testing_main.py --trial_number <an integer> --model <model name>`

Argument options:

* `trial_number`: the trial number corresponding to the training trial you want to test

* `model`: `deep_residual_network`,`deep_residual_network_rednet`,`deep_residual_network_SRCNN`,`rednet`,`SRCNN`,`VDSR`

* `psnr_only` (optional): either `true` or `false`. Compute psnr only, without saving resulting images.

Please note the following:
* The test model must be the same as the training model with that trial number
* The result images and the files `test_result_psnr.txt` and `test_result_paths.txt` will be stored in a new folder called `CUB200_outs` under `data`. Before running a new test, you must manually remove any old `CUB200_outs` from the `data` folder.

### Results
#### Models trained with uncropped input images
![image](https://drive.google.com/uc?export=view&id=1uoXgXllUtUV4noIdAv0GwHrjVYyTR6xd)
![image](https://drive.google.com/uc?export=view&id=1_-07H-6MqyJovf0r7OWW-nEvsiFz6Jwi)

#### Models trained with cropped input images
![image](https://drive.google.com/uc?export=view&id=1S6hSn3hIOma7O79hom94ASt9vemvAKn8)
![image](https://drive.google.com/uc?export=view&id=1iRECSJVTWgyF6t9BsCoRevR5bztnZZ6R)
