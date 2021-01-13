## Image classification: ##
Image classification is a supervised learning problem. We define a set of target classes (in our case Disney Cartoons), and train a model to recognize them using labeled example photos. In here, we make use of [TensorFlow 2.x](https://www.tensorflow.org/guide/effective_tf2) and [Keras](https://keras.io/) in order to build, train, and optimize our model.

## Dataset ##
we used a pretrained model. In detail, we build a base model from the MobileNet V2 model developed at Google. This is pre-trained on the ImageNet dataset, a large dataset consisting of 1.4M images and 1000 classes. ImageNet is a research training dataset with a wide variety of categories like jackfruit and syringe. Based on this base-model, we will add our classification layer for the Disney princesses. The outcome will be a Convolutional Neural Network (CNN). 

