## Image classification: ##
Image classification is a supervised learning problem. We define a set of target classes (in our case Disney Cartoons), and train a model to recognize them using labeled example photos. In here, we make use of [TensorFlow 2.x](https://www.tensorflow.org/guide/effective_tf2) and [Keras](https://keras.io/) in order to build, train, and optimize our model.

## Dataset ##
we used a pretrained model. In detail, we build a base model from the [MobileNetV2](https://ai.googleblog.com/2018/04/mobilenetv2-next-generation-of-on.html) model developed at Google. This is pre-trained on the ImageNet dataset, a large dataset consisting of 1.4M images and 1000 classes. ImageNet is a research training dataset with a wide variety of categories like jackfruit and syringe. Based on this base-model, we will add our classification layer for the Disney princesses. The outcome will be a Convolutional Neural Network (CNN). 

Our dataset containing 58 cartoon characters and 406 images (348 images for training which each character containing 6 images and 58 images for validation which each character containing 1 image).

## Code Explanation ##

### Required dependencies ###
  * TensorFlow:
    * is a free and open-source software library for dataflow and differentiable programming across a range of tasks. It is a symbolic math library, and is also used for machine learning applications such as neural networks.
  * Keras: 
    * is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research. 
  * Numpy: 
    * is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays. 
  * MatPlotLib: 
    * is a plotting library for the Python programming language and its numerical mathematics extension NumPy. It provides an object-oriented API for embedding plots into applications using general-purpose GUI toolkits like Tkinter, wxPython, Qt, or GTK+. 
  * os: 
    * This module provides a portable way of using operating system dependent functionality.
    
 ### Training and Validation data
   * we specify the training and validation directory. This can easily be done by extracting the folder names of the dataset.
    * train
    * validation
 ### Set up our variables ###
  * As we are only having a small dataset and try not to overfit our model right away - I recommend going for a max. of 50 epochs with a batch size of 5. In addition, we will rescale our pictures to 150x150 pixels as we plan to use a 1D [150,150] Tensor.
 ### Data preparation ###
 * We format the images into appropriately pre-processed floating point tensors before feeding to the network. Therefore, we decode contents of these images and convert it into proper grid format as per their RGB content.
 * convert them into floating point tensors. Finally, we will rescale the tensors from values between 0 and 255 to values between 0 and 1, as neural networks prefer to deal with small input values (using `ImageDataGenerator` class provided by `tf.keras`)
 * We have 58 classes in our dataset
 
### Model ###
* As mentioned above, we will leverage the pretrained [MobileNetV2](https://ai.googleblog.com/2018/04/mobilenetv2-next-generation-of-on.html) model by Google for our base model, due to the fact that we are not having a vast amount of training data for our Disney Cartoons. This approach is called transfer learning and is especially valuable in cases where not much data is present for training purposes.
* As we do not want to retrain the base model (MobileNetV2), we gonna exclude it from the training process by setting the trainable argument to false.
* We  build a GlobalAveragePooling layer on top of the base model. Remember, we currently have a feature output share of (1, 5, 5, 1280). However, for classification of 58 classes, we just want to have a (1, 58) Tensor.
* Next, compiling our model. For our model, we choose the ADAM optimizer and categorical cross entropy loss function.
     
## Results ##
![picture alt](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/cartoon-classification/Results/Hiro_match2.png "Title is optional")
 ![picture alt](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/cartoon-classification/Results/merida_match.png "Title is optional")
 ![picture alt](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/cartoon-classification/Results/moona_match.png "Title is optional")
 ![picture alt](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/cartoon-classification/Results/moona_match2.png "Title is optional")
 ![picture alt](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/cartoon-classification/Results/russe_match2.png "Title is optional")
 ![picture alt](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/cartoon-classification/Results/alice_match.png "Title is optional")
 ![picture alt](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/cartoon-classification/Results/eric_match.png "Title is optional")
 ![picture alt](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/cartoon-classification/Results/flynn_match.png "Title is optional")
 ![picture alt](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/cartoon-classification/Results/images_match.png "Title is optional")
 ![picture alt](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/cartoon-classification/Results/rapunzel.png "Title is optional")
  ![picture alt](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/cartoon-classification/Results/slivermist_match.png "Title is optional")
   ![picture alt](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/cartoon-classification/Results/snow_match.png "Title is optional")
   ![picture alt](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/cartoon-classification/Results/hiro_match.png "Title is optional")


