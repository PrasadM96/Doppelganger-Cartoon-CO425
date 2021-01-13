# Pretrained Comparison model #

## Method Explanantion ##

### Face detection ###
* Face detection in human images - MTCNN
* Face detection in cartoon images - Dlib cnn_face_detection_model_v1

### Feature Extraction ###
* Pre-trained model VGG16 is used for feature extraction from images.
VGG16 is a convolution neural net (CNN ) architecture which was used to win ILSVR(Imagenet) competition in 2014. It is considered to be one of the excellent vision model architecture till date. Most unique thing about VGG16 is that instead of having a large number of hyper-parameter they focused on having convolution layers of 3x3 filter with a stride 1 and always used the same padding and max pool layer of 2x2 filter of stride 2. It follows this arrangement of convolution and max pool layers consistently throughout the whole architecture. In the end, it has 2 FC(fully connected layers) followed by a softmax for output. The 16 in VGG16 refers to it has 16 layers that have weights. This network is a pretty large network and it has about 138 million (approx) parameters.

### Comparision ###
* Feature vectors are compared using euclidean distance.

### Results ###
