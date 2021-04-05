# ETH-XGaze baseline
Official implementation of ETH-XGaze dataset baseline.

## ETH-XGaze dataset
ETH-XGaze dataset is a gaze estimation dataset consisting of over one million high-resolution images of varying gaze under extreme head poses. We established a simple baseline test on our ETH-XGaze dataset and other datasets. This repository includes the code and pre-trained model. Please find more details about the dataset on our [project page](https://ait.ethz.ch/projects/2020/ETH-XGaze/).

## License
The code is under the license of [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## Requirement
* Python 3.5
* Pytorch 1.1.0, torchvision
* opencv-python
#### For model training
* h5py to load the training data
* configparser
#### For testing
* dlib for face and facial landmark detection.

## Training
- You need to download the [ETH-XGaze](https://ait.ethz.ch/projects/2020/ETH-XGaze/) dataset for training. After downloading the data, make sure it is the version of pre-processed 224*224 pixels face patch. Put the data under '\data\xgaze'
- Run the `python main.py` to train the model
- The model will be saved under 'ckpt' folder.

## Test
The demo.py files show how to perform the gaze estimation from input image. The example image is already in 'example/input' folder.
- First, you need to download the [pre-trained model](https://drive.google.com/file/d/1Ma6zJrECNTjo_mToZ5GKk7EF-0FS4nEC/view?usp=sharing), and put it under "ckpt" folder.
- And then, run the 'python demo.py' for test.

## Data normalization
The 'normalization_example.py' gives the example of data normalization from the raw dataset to the normalized data.

## Citation
If using this code-base and/or the ETH-XGaze dataset in your research, please cite the following publication:

    @inproceedings{Zhang2020ETHXGaze,
      author    = {Xucong Zhang and Seonwook Park and Thabo Beeler and Derek Bradley and Siyu Tang and Otmar Hilliges},
      title     = {ETH-XGaze: A Large Scale Dataset for Gaze Estimation under Extreme Head Pose and Gaze Variation},
      year      = {2020},
      booktitle = {European Conference on Computer Vision (ECCV)}
    }

## FAQ
**Q: Where are the test set labels?**<br/>
You can submit your test result to our leaderboard and get the results. Please do follow the registration first, otherwiese, your request will be ignored. [Link to the leaderboard](https://competitions.codalab.org/competitions/28930).

**Q: What is the data normalization?**<br/>
As we wrote in our paper, data normalization is a method to crop the face/eye image without head rotation around the roll axis. Please refer to the following paper for details: [Revisiting Data Normalization for Appearance-Based Gaze Estimation](https://www.perceptualui.org/publications/zhang18_etra.pdf)

**Q: Why convert 3D gaze direction (vector) to 2D gaze direction (pitch and yaw)? How to convert between 3D and 2D gaze directions?**<br/>
Essentially to say, 2D pitch and yaw is enough to describe the gaze direction in the head coordinate system, and using 2D instead of 3D could make the model training easier. There are code examples on how to convert between them in the "utils.py" file as `pitchyaw_to_vector` and `vector_to_pitchyaw`.



