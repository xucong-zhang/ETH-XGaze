# ETH-XGaze baseline
Official implementation of ETH-XGaze dataset baseline. This repository is not completed yet.

## ETH-XGaze dataset
ETH-XGaze dataset is a gaze estimation dataset consisting of over one million high-resolution images of varying gaze under extreme head poses. We established a simple baseline test on our ETH-XGaze dataset and other datasets. This repository includes the code and pre-trained model. Please find more details about the dataset on our [project page](https://ait.ethz.ch/projects/2020/ETH-XGaze/).

## License
The code is under the license of [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## Requirement
Pytorch 1.1.0
Python 3.5

## Use
Download the repository
### Training
You need to download the ETH-XGaze dataset for training

### Test
The demo.py files show how to perform the gaze estimation from input face image.
You additionally need the face-alignment package using "pip install face-alignment".
You need to download the [pre-trained model](https://drive.google.com/file/d/1Ma6zJrECNTjo_mToZ5GKk7EF-0FS4nEC/view?usp=sharing), and put it under "ckpt" folder.

