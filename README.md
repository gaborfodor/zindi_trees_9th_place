# zindi_trees_9th_place


### Install dependencies
create virtualenv with python3.8

```
pip install --upgrade pip

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

pip install -r requirements.txt
```


### Data preparation
Download all the images to a single directory (e.g. `./data`).
Modify wandb user and data absolute path in `src/configs/default_config.py`.

### train models
Train five `efficientnetv2_rw_s` models with 5 fold CV.

`cd src`
`./train_models.sh`

### Create submission
Just blend the predictions.
