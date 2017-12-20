# **Traffic Sign Recognition** 


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/AliBaheri/Traffic_Sign_Classifier_SDCND/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is `34799`
* The size of the validation set is `4410`
* The size of test set is `12630`
* The shape of a traffic sign image is `(32,32,3)`
* The number of unique classes/labels in the data set is `43`

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set:

<img width="500" src="https://github.com/AliBaheri/Traffic_Sign_Classifier_SDCND/blob/master/Results/NumberofAccurences.png"> 


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

* Normalize images according to `image - image.mean()/image.std()`.
(After a couple of trial and errors, I decided to choose this criteria rather than `image - 128/128`.
* Convert RGB images to gray images using OpenCV library.
* Resize images to `(32,32)`

The above (simple) pipeline for prepreosseing the image data ensure us that training inputs follow a fairly similar distribution for each feature over the course of learning.
 

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

Here is the final architecture of my model:

| Layer         	|     Description	        		| 
|:---------------------:|:---------------------------------------------:| 
| Input         	| 32x32x1 RGB image   				| 
| Convolution 2d     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| relu 			| activation function				|
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x6 	|
| Convolution 2d	| 1x1 stride, valid padding, outputs 10x10x16   |
| relu			| activation function		 		|
| Max pooling	      	| 2x2 stride, valid padding, outputs 5x5x16 	|
| Flatten		| 5x5x16 -> 400x1				|
| Fully connected	| 400x1 -> 120x1				|
| relu 			| activation function		 		|
| Dropout		| keep_prob=0.5				 	|
| Fully connected	| 120x1 -> 84x1					|
| tanh 			| activation function		 		|
| Dropout		| keep_prob=0.5				 	|
| Softmax		| 84x1 -> 43x1					|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

| Parameter      	|  Setting	| 
|:---------------------:|:-------------:| 
| EPOCHS         	|  	60	| 
| BATCH_SIZE    	|  	100	| 
| LEARNING_RATE  	|  	0.001	|
| KEEP_PROB        	|  	0.5	|


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I started with basic LeNet architecture and it turned out that after a couple of modifications we are able to achieve above `93%` accuracy in validation set (my validation set accuracy was `94%`). The following items were modified from basic LeNet architecture:

* Adding dropout (to prevent from overfitting)
* Replace `tanh` in the last layer rather than `relu` (I did a couple of experiments and I observed slightly better performance using `tanh` activation function. However, it should be emphasized that to achieve significantly better results we need to resort to more sophisticated data preprocessing techniques and models.)

My final model results were:
* training set accuracy of `98.6%`
* validation set accuracy of `94%`
* test set accuracy of `91.7%`

## Test a Model on New Images

I have downloade 10 images from the web to test the overall performance of my model. It turns out for the the model is able to classify 6 out of 10 images correctly:


| New image from the web      		|  Prediction									| 
|:---------------------:|:---------------------------------------------:|	 
| General caution  | General caution    							|  
| No entry   	| No entry			|
| Stop 			    | End of no passing by vehicles over 3.5 metric tons										|
| Priority road 	| Priority road							|
| Speed limit (20km/h) 		| Speed limit (20km/h)									|
| Turn left ahead  | Beware of ice/snow    							|  
| Slippery road  	| Slippery road			|
| Double curve 			    | Children crossing										|
| Yield 	| Yield							|
| Ahead only 		| General caution									|

Among those predicted incorrectly by model, I observed that the image `2` and image `10` suffer from the poor quality. That's why the model unable to successfully classify them. On the other hands, there are a coplue of images (such as image `1` and image `9`) which are crystal clear. Thus, it is easy for the classifier to correctly classify them.

One can conclude from this result that further data preprocessing techniques such as data augmentation or even spatial transformation are needed for better prediction. Furthermore, more sophisticated architecture such as VGG-net or inception model can be utilized to achieve higher prediction accuracy.

* Question: Discuss how certain or uncertain the model is for its predictions?
