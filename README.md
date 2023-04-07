# BridgeHand2Vec

## Installation

This was developed & tested on python3.8.10, other popular python versions should work as well.

```
pip3 install -r requirements.txt
```

## Repository contents

* BridgeHand2Vec model
* [Notebook showing sample usage of the model](sample_usage.ipynb)
* Training script and (sample) training data

## Training

```
python3 train.py
```

Training should take few minutes, run `tensorboard --logdir runs/vectors/` to watch progress.
Training data in the `data` directory is only a part of the data `model.pth` was trained on,
so new model will be much less accurate.
