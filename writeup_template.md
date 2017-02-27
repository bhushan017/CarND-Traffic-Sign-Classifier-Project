#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyse the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./test_images/12.jpg "Traffic Sign 1"
[image5]: ./test_images/33.jpg "Traffic Sign 2"
[image6]: ./test_images/34.jpg "Traffic Sign 3"
[image7]: ./test_images/39.jpg "Traffic Sign 4"
[image8]: ./test_images/40.jpg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/bhushan017/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the 2nd code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the 3rd code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart with the number of inputs per class.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.
The code for this step is contained in the fourth code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because it will much easier to train the model. For this particular task of classifying traffic sign colour images is not necessary. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data to achieve consistency in dynamic range for all the images. 

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for training and validation sets is contained in the 2nd code cell of the IPython notebook.  

Number of training examples = 34799
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43
Number of validation examples = 4410

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the 8th cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,valid padding, outputs 14x14x32 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x64		|
| RELU					|												|
| Max pooling	      	| 2x2 stride,valid padding, outputs 5x5x64 				|
| Fully connected		| Input = 1600. Output = 600					|
| RELU					|												|
| Dropout					|												|
| Fully connected		| Input = 600. Output = 150					|
| RELU					|												|
| Dropout					|												|
| Fully connected		| Input = 150. Output = 43					|
|						|												|


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 8th cell of the ipython notebook. 

To train the model, I used AdamOptimizer for optimizing. 

Batch size = 128

Epochs     = 20 

mu         = 0

signal     = 0.1

learning rate = 0.001
####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 99.9%
* validation set accuracy of 95.6%
* test set accuracy of 94.6%

If a well known architecture was chosen:
* What architecture was chosen? 

 LeNet Architecture. 
* Why did you believe it would be relevant to the traffic sign application?

 LeNet is good at image classification. We are classifying traffic sign images which makes the Lenet architecture ideal for this project.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

 I am plotting the cross entropy loss during my evaluation, I see the loss is reducing smoothly. This shows that the model is working well.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 18th cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Priority road      		| Priority road   									| 
| Turn right ahead     			| Turn right ahead										|
| Turn left ahead					| Turn left ahead											|
| Keep left	      		| Keep left					 				|
| Roundabout mandatory			| Roundabout mandatory    							|

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 18th cell of the Ipython notebook.


 For the first image, the model is sure that this is a priority road (probability of 0.99), and the image does contain a Priority road. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         | Priority road   									| 
| .01     				| Yield 										|
| .00			      |Roundabout mandatory										|
| .00	      		| End of no passing by vehicles over 3.5 metric tons					 				|
| .00				     | End of no passing      							|

For the second image, the model is sure that this is a Turn right ahead (probability of 1), and the image does contain a Turn right ahead. The top five soft max probabilities were

| Probability          |     Prediction              | 
|:---------------------:|:---------------------------------------------:| 
|1   | Turn right ahead            | 
|.00 | Traffic signals         |
|.00 |Keep left          |
|.00 | Right-of-way at the next intersection          |
|.00 | Speed limit (20km/h)             |

For the thrid image, the model is sure that this is a Turn left ahead (probability of 0.99), and the image does contain a Turn left ahead. The top five soft max probabilities were

| Probability          |     Prediction              | 
|:---------------------:|:---------------------------------------------:| 
| .99          | Turn left ahead            | 
| .01          | Ahead only         |
| .00          |Roundabout mandatory         |
| .00          |Keep right          |
| .00          | Priority road            |

For the fourth image, the model is sure that this is a Keep left (probability of 0.99), and the image does contain a Keep left. The top five soft max probabilities were

| Probability          |     Prediction              | 
|:---------------------:|:---------------------------------------------:| 
| .99          | Keep left            | 
| .01          | Speed limit (50km/h)         |
| .00          |Speed limit (60km/h)         |
| .00          |Turn right ahead      |
| .00          | Children crossing           |

For the fifth image, the model is sure that this is a  Roundabout mandatory  (probability of 1), and the image does contain a  Roundabout mandatory. The top five soft max probabilities were

| Probability          |     Prediction              | 
|:---------------------:|:---------------------------------------------:| 
|   1          | Roundabout mandatory            | 
| .00        | Priority road        |
| .00        |Speed limit (20km/h)        |
| .00        |Right-of-way at the next intersection    |
| .00        | Go straight or left           |
