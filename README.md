# BridgeHand2Vec

## Installation

We used `python3.8.10`, other popular python versions should work as well.

```
pip3 install -r requirements.txt
```

## Repository contents

* BridgeHand2Vec model
* [Notebook showing sample usage of the model](sample_usage.ipynb)
* Training script and (sample) training data

## Training

Run training on a small subset of the training data:

```
python3 train.py
```

This should take few minutes, run `tensorboard --logdir runs/vectors/` to watch the progress.

To train on the full dataset, unzip `data/vectorsnet_train.zip` file (e.g. `unzip data/vectorsnet_train.zip`) and change filenames in `train.py` script.
