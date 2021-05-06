# Demosaicing with Deep Learning
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
Run `preprocess.py`

### Training
Run the command `train_main.py --resume_from_ckp <"true" or "false"> --trial_number <an integer> --model <model name> --num_epochs <an integer> --lr <a float number>`

Argument options:

* `resume_from_ckp`: `true`, `false`

* `trial_number`: the trial number

* `model`: `deep_residual_network`,`deep_residual_network_rednet`,`deep_residual_network_SRCNN`,`rednet`,`SRCNN`

* `num_epochs`: number of epochs

* `lr` (optional): between `1e-6` and `1e-1`, defaults to `1e-4` if not given

### Testing
Run the command `testing_main.py --trial_number <an integer> --model <model name>`

Argument options:

* `trial_number`: the trial number corresponding to the training trial you want to test

* `model`: `deep_residual_network`,`deep_residual_network_rednet`,`deep_residual_network_SRCNN`,`rednet`,`SRCNN`

Please note the following:
* The test model must be the same as the training model with that trial number
* The result images will be stored in a new folder called `CUB200_outs` under `data`. Before running a new test, you must manually remove any old `CUB200_outs` from the `data` folder.

