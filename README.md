# zindi_trees_9th_place


### Install dependencies
create virtualenv with python3.8

```
pip install --upgrade pip

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

pip install -r requirements.txt
```


### Data preparation
Download all the images to `./data`.

Modify wandb user and data path in `src/configs/default_config.py`.

### Train models
Train five `efficientnetv2_rw_s` models with 5 fold CV.

`cd src`
`./train_models.sh`

### Create submission
Just blend the predictions.

`python submit.py`

Submit `subs/e5_25.csv` to Zindi it should give 1.608xxx on the private LB. 

### Acknowledgements
Thanks for Pascal Pfeiffer and Philipp Singer for sharing their solution https://github.com/pascal-pfeiffer/kaggle-rsna-2022-5th-place
Their framework was really useful and it was easy to simplify to this image regression problem.
