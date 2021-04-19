# deep_learning_for_demosaicing
To obtain the mcmaster dataset, following the following steps:
1. Create `data` folder if it does not exist yet
2. Run `wget https://www4.comp.polyu.edu.hk/~cslzhang/DATA/McM.zip -P "./data"`
3. Run `unzip -P McM_CDM ./data/McM.zip -d "./data"`
4. Run `rm ./data/McM.zip`

Before running `training.py`:
1. Remember to manually change the following in `training.py` at the top of the script (ideally command line argument will be implemented later): resume_from_ckp, trialNumber, num_epochs

Before running `testing.py`:
1. Remember to manually change the following in `testing.py` at the top of the script (ideally command line argument will be implemented later): trialNumber
