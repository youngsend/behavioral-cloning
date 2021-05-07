# Behavioral Cloning Project

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

## Preparation

### 1. simulator

- I download the simulator of `Term 1, Version 2, Linux` from [simulator repository](https://github.com/udacity/self-driving-car-sim).

#### 2. problem solved: car not move in autonomous mode 

- I encountered the problem that ego vehicle can not move during autonomous mode, and the problem is conda environment (especially io packages) I've been used. 
  - I created a new environment with this [yaml file](behav-clone-env.yaml) which I copied from [here](https://knowledge.udacity.com/questions/411931): `conda env create --file behav-clone-env.yaml`. 
  - and installed torch with `conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch`(because I use PyTorch, and my CUDA version is `10.2.89`), and finally succeeded in making ego vehicle move.

# Files & code

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

- **This project is implemented in `PyTorch`**.

- My project includes the following files:
  - `model.py`: containing the definition of the model.
  - `dataset.py`: containing the custom dataset class, used by dataloader.
  - `util.py`: containing model training and validation.
  - `main.py`: containing the training process used to produce `model.pth`
  - `drive.py`: for driving the car in autonomous mode.
  - `model.pth`: containing a trained convolution neural network.
  - `README.md`: summarizing the results.

#### 2. Submission includes functional code

- Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing (**PyTorch is needed in conda environment**).

  ```bash
  python drive.py model.pth
  ```

- **If PyTorch is not installed in conda environment, this [video](output_video.mp4) is available to show the result**.

#### 3. Submission code is usable and readable

- The **`main.py`** file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.
- The `dataset.py` file defines a **custom dataset**, which is used to initialize train and validation dataloader. **The reading of images is left in `__getitem__()` so that all images are not stored in the memory at once but read as required**. 
  - I referred to this [tutorial article](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html) to write custom dataset for this project.

### Model architecture and training strategy

#### 1. An appropriate model architecture has been employed

- the model definition is written in `model.py`.

- The model layers are: 

  ```bash
  LeNetRevised(
    (conv1): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
    (bn1): BatchNorm2d(6, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
    (bn2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (fc1): Linear(in_features=20944, out_features=120, bias=True)
    (fc2): Linear(in_features=120, out_features=1, bias=True)
  )
  ```

- `BatchNorm2d` layers are used to reduce overfitting.

- `relu` are used to introduce nonlinearity (code line 16 in **`forward()`** function). `relu` is not shown above because in pytorch it's usually written in `forward()` function, so as  `max_pool`.
- Since I used `PIL.Image` to read image, pixel value has been in `[0, 1]`. So I do not need to divide them by 255.

#### 2. Attempts to reduce overfitting in the model

- The model contains **`BatchNorm2d`** layers in order to reduce overfitting (`model.py` lines 9, 11). 

- The model was trained and validated on different data sets to ensure that the model was not overfitting (`main.py` code line 23-28). 

  ```python
  dataset = BehaviorCloneDataset(csv_file='data/new_data/driving_log.csv', root_dir='data/new_data', transform=transform)
  n_samples = len(dataset)
  
  # split train and validation data
  train_size = int(n_samples * 0.9)
  val_size = n_samples - train_size
  train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
  train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
  ```

- The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

- The model used an adam optimizer, and I used the default learning rate of `1e-3` (`main.py` line 25).

  ```python
  optimizer = optim.Adam(model.parameters(), lr=1e-3)
  ```

- Although it's said that `Adam` optimizer does not need a learning rate, learning rate parameter does exist in pytorch `Adam` initializer.

#### 4. Appropriate training data

- training data was chosen to keep the vehicle driving on the road.
- I drove two laps on track 1 using center lane driving.
- I used a combination of images taken by center, left and right camera when driving in lane center.

### Architecture and training

#### 1. Solution Design Approach

- My first step was to use a convolution neural network model similar to the traffic sign classifier (**change the final output dimension from 43 to 1**). I thought this model might be appropriate because the **lane boundary curve** should be the most important pattern to detect. Since lane boundary curve is **edge**, it's not that difficult image pattern.

- In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that the model had a low mean squared error on both the training set and validation set.

- However the vehicle can not drive on lane center, and fell off the track soon. In order to deal with this problem, I mainly tried several methods to augment the training data, which is described later.

#### 2. Final Model Architecture

- The final model architecture (`model.py`) consisted of a convolution neural network with the following layers and layer sizes: ![](/home/sen/senGit/behavioral-cloning/examples/nn.svg)
  - `80x320` is the cropped image size.
  - the correct flattened size should be `1x20944`. `1x320` is plotted because the visualization can not show the large size as `20944`.
  - Since the flattened size is too large, maybe it's better to add more convolution and max-pooling layer.

#### 3. Creation of the Training Set & Training Process

- I only recorded two laps on track 1 using center lane driving. Here is an example image of center lane driving:![](/home/sen/senGit/behavioral-cloning/examples/center.png)

- The preprocessing I employed includes:

  - cropping image so that only the road part is left: ![](/home/sen/senGit/behavioral-cloning/examples/cropped.png)

  - randomly horizontally flipped images and angles thinking that this would mitigate the bias of steer angle because I only recorded counter clock driving. For example, here is an image that has then been flipped: ![](/home/sen/senGit/behavioral-cloning/examples/hflipped.png)

  - use left and right camera images too, and I used a **steer angle correction of `0.1`**. This is the ultimate method that made my model work.

    ```python
            if column_index == 1:
                # left image
                steer = steer + self.steer_correction
            elif column_index == 2:
                # right image
                steer = steer - self.steer_correction
    ```

- I collected `20898` training data points and `2322` validation data points (`10%`).

- I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The number of epochs was `10` as evidenced by the training log:

  ```bash
  2021-05-07 15:38:45.159645 Epoch 1, Training loss 1.2237834177615081
  val loss: 0.022784347419400473
  2021-05-07 15:39:16.468751 Epoch 2, Training loss 0.019918042989817964
  val loss: 0.021021103561931365
  2021-05-07 15:39:47.449178 Epoch 3, Training loss 0.018894255841976824
  val loss: 0.019294310738710133
  2021-05-07 15:40:18.104617 Epoch 4, Training loss 0.01870679548323519
  val loss: 0.01931086723768228
  2021-05-07 15:40:48.382019 Epoch 5, Training loss 0.018288318201648897
  val loss: 0.018573566465764434
  2021-05-07 15:41:18.998145 Epoch 6, Training loss 0.01780689003042854
  val loss: 0.019609074876014446
  2021-05-07 15:41:49.966874 Epoch 7, Training loss 0.01763586824138959
  val loss: 0.018770636084514694
  2021-05-07 15:42:20.729795 Epoch 8, Training loss 0.017481831870745264
  val loss: 0.01825996021412917
  2021-05-07 15:42:51.226614 Epoch 9, Training loss 0.017147911555882807
  val loss: 0.018614856803135293
  2021-05-07 15:43:21.705599 Epoch 10, Training loss 0.01713934335971553
  val loss: 0.0188548816745547
  ```

  - I find that the beginning of the training has a big impact on the final training loss and validation loss. Sometimes the training loss and validation loss may be higher and do not decrease at all.

## Video creation

#### 1. Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### 2. `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.
