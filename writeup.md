# **Behavioral Cloning**

The goals / steps of this project were the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results in this file


[//]: # (Image References)

[image1]: ./figures/cropped_images.png "Visualizing cropped images"
[image2]: ./figures/original_and_cropped.png "Visualizing original and cropped images"
[image3]: ./figures/model_architecture.png "Model"
[image4]: ./figures/angles_histogram_original_dataset.png "original dataset histogram"
[image5]: ./figures/angles_histogram_after_repopulating_dataset.png "re-populated dataset histogram"
[image6]: ./figures/plot_loss_and_val.png "plot loss and val"
[image7]: ./figures/Nvidia_model.png "Nvidia model"



---

#### My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* process and resulrs are summarized herein
* video.mp4 (48 fps) showing my model successfully driving the simulator for two full laps without leaving the road

#### comments over code in this repo
The model provided in this repo can be used to successfully operate the simulation and complete at least two full laps without leaving the road.

* The code in model.py uses a Python generator to generate data for training. 
* The model.py file contains the code for training and saving the trained model. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.
* The model.py code is clearly organized and commented.

### Model Architecture
Architecture chosen similar to Nvidia proven model, with minor twiks as required. discussed further later in this doc.

![alt text][image7]

This is the layout of my final model -

![alt text][image3]

#### short discussion about layers chosen -
* first layer is cropping irrelevant pixels (50 pixels from top and 10 pixels from bottom), to reduce amount of parameters in the model
* second layer is regularization and mean centering the pixel data, to achieve faster more efficient training
* third layer is convolution (5 by 5 kernel) with padding and relu activation for non-linearity
* forth layer is maxpool with padding, to reduce amount of parameters in the model
* fifth layer is 10% dropout, to avoid over fitting

* layers 3-5 now repeted 3 more times (with growing capacity of neurons, as seen in the image above)

* finally 3 fully connected layers to produc one float type output

image input samples were plotted (original input shape: (160, 320, 3), after cropping shape: (100, 320, 3)) -

![alt text][image1]

#### model training & means to avoid overfitting

The model contains 4 dropout layers (p=10%) in order to reduce overfitting
Additionally the model was only trained for 6 EPOCHs in total, again to avoid overfitting.
Adagrad optimizer was chosen after experimenting with different optimizers. 
prameters that worked best for me are as following lr=0.001, epsilon=1e-08, decay=1e-06.

![alt text][image6]

It is apparent from plotting training loss and validation loss that over fitting was avoided, in theory additional training may have resulted in a better model, however for lack of time and computing power I have decided to stop here and move to the next project.

#### training and validation data

I have collected data from the two different simulators in order to generalize better. I have collected as many images as I could driving both tracks in both directions.

The data was split for training set 85% and validation set 15%.

number of samples in train set: 22162
number of samples in validation set: 3912

training set was shuffled to support a more efficient / smooth training.

image samples were plotted to verify quality and to decide how the images should be cropped best -

![alt text][image2]


plotting the histogram data based on steering angle, reveals a disturbing imbalance in the data. see below -

![alt text][image4]

the strong bias towards zero angle was minimized by re-populating the training data by increasing frequency of samples as much as abs(steering_angle) is bigger (validation data, unchanged!!!)

initial dataset size:  22162
total samples after re_dist: 79594

this is the distribution of the re-populated training data set -

![alt text][image4]


No test set. model trained was tested driving the simulator.
Very fun project. loved it.
