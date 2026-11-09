# UrbanSound8K CNN Audio Classification

A PyTorch project that classifies environmental audio recordings into the 10 categories of the UrbanSound8K dataset. Audio is converted to Mel spectrograms and processed by a convolutional neural network.

This project is inspired by Valerio Velardo's **PyTorch for Audio + Music Processing** course, available on YouTube. It is an independent educational project and is not affiliated with the course or its author.

## Requirements

- Python 3
- PyTorch and the matching `torchaudio` version. Install them using the instructions for your CPU or CUDA setup on the [official PyTorch website](https://pytorch.org/get-started/locally/).
- `pandas` and `torchsummary`

For example, after installing PyTorch and `torchaudio`:

```bash
pip install pandas torchsummary
```

## Dataset

Download the UrbanSound8K dataset and arrange it as follows:

```text
UrbanSound8K/
├── audio/
│   ├── fold1/
│   ├── fold2/
│   └── ...
└── metadata/
	└── UrbanSound8K.csv
```

The dataset path defaults to `../datasets/UrbanSound8K/`. Update `DATASET_PATH` in `urbansounddataset.py` if your copy is stored elsewhere.

## Train

From the project directory, run:

```bash
python train.py
```

Training automatically uses CUDA when available and otherwise uses the CPU. The model weights are saved to `feedforwardnet.pth`.

## Run Inference

After training, run:

```bash
python inference.py
```

The script loads `feedforwardnet.pth`, runs inference on the first sample in the dataset, and prints the predicted and expected class labels.

## Audio Classes

The model predicts one of these UrbanSound8K classes:

`air_conditioner`, `car_horn`, `children_playing`, `dog_bark`, `drilling`, `engine_idling`, `gun_shot`, `jackhammer`, `siren`, `street_music`.
