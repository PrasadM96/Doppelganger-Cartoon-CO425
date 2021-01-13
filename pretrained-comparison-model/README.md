# Pretrained Comparison model #

## Method Explanantion ##

### Face detection ###
* Face detection in human images - MTCNN
* Face detection in cartoon images - Dlib cnn_face_detection_model_v1

### Feature Extraction ###
* Pre-trained model VGG16 is used for feature extraction from images.
VGG16 is a convolution neural net (CNN ) architecture which was used to win ILSVR(Imagenet) competition in 2014. It is considered to be one of the excellent vision model architecture till date. Most unique thing about VGG16 is that instead of having a large number of hyper-parameter they focused on having convolution layers of 3x3 filter with a stride 1 and always used the same padding and max pool layer of 2x2 filter of stride 2. It follows this arrangement of convolution and max pool layers consistently throughout the whole architecture. In the end, it has 2 FC(fully connected layers) followed by a softmax for output. The 16 in VGG16 refers to it has 16 layers that have weights. This network is a pretty large network and it has about 138 million (approx) parameters.

### Comparision ###
* Feature vectors are compared using euclidean distances.

      
## Results ##

<div style="height:300">
<img src="https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/pretrained-comparison-model/Results/Alice/Alice_real1.jpg" width=200 alt="accessibility text">
 <span> <img align="right" src="https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/pretrained-comparison-model/Results/Alice/alice_match_2.png"> </span>
</div>
<hr>
<div style="height:300" >
<img src="https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-shape-predictor-human-human-mappring/Test%20images/elarstic.jpg"  width=200 alt="accessibility text">
 <span> <img align="right" src="https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-shape-predictor-human-human-mappring/Results/elarstic_match.png"> </span>
</div>
<hr>
<div style="height:300">
<img src="https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-shape-predictor-human-human-mappring/Test%20images/eric.jpg" width=200  alt="accessibility text">
  <span> <img  align="right" src="https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-shape-predictor-human-human-mappring/Results/eric_match.png"> </span>
</div>
<hr>
<div style="height:300">
<img src="https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-shape-predictor-human-human-mappring/Test%20images/flynn.jpg" width=200  alt="accessibility text">
  <span> <img align="right" src="https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-shape-predictor-human-human-mappring/Results/flynn_match.png"> </span>
</div>
<hr>
<div style="height:300">
<img src="https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-shape-predictor-human-human-mappring/Test%20images/hans.jpg" width=200  alt="accessibility text">
  <span> <img align="right" src="https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-shape-predictor-human-human-mappring/Results/hans_match.png"> </span>
</div>
<hr>
<div style="height:300">
<img src="https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-shape-predictor-human-human-mappring/Test%20images/hiccup.jpg" width=200  alt="accessibility text">
  <span> <img align="right" src="https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-shape-predictor-human-human-mappring/Results/hiccup_match.png"> </span>
<hr>
<div style="height:300">
<img src="https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-shape-predictor-human-human-mappring/Test%20images/hiro.jpg" width=200  alt="accessibility text">
  <span> <img align="right" src="https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-shape-predictor-human-human-mappring/Results/hiro_match.png"> </span>
</div>
<hr>
<div style="height:300">
<img src="https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-shape-predictor-human-human-mappring/Test%20images/images.jpg" width=200  alt="accessibility text">
  <span> <img align="right" src="https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-shape-predictor-human-human-mappring/Results/images_match.png"> </span>
</div> 
<hr>
<div style="height:300">
<img src="https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-shape-predictor-human-human-mappring/Test%20images/jackfrost_2.jpg" width=200  alt="accessibility text">
  <span> <img align="right" src="https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-shape-predictor-human-human-mappring/Results/alice_match.png"> </span>
</div> 
<hr>
<div style="height:300">
<img src="https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-shape-predictor-human-human-mappring/Test%20images/rapunzel.jpg" width=200  alt="accessibility text">
  <span> <img align="right" src="https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-shape-predictor-human-human-mappring/Results/rapunzel_match.png"> </span>
</div> 
<hr>
<div style="height:300">
<img src="https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-shape-predictor-human-human-mappring/Test%20images/sid.jpg" width=200  alt="accessibility text">
  <span> <img align="right" src="https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-shape-predictor-human-human-mappring/Results/sid.png"> </span>
</div>
<hr>
 <div style="height:300">
<img src="https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-shape-predictor-human-human-mappring/Test%20images/slivermist.jpg" width=200  alt="accessibility text">
   <span> <img align="right" src="https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-shape-predictor-human-human-mappring/Results/silvermist.png"> </span>
</div> 
<hr>
<div style="height:300">
<img src="https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-shape-predictor-human-human-mappring/Test%20images/snow.jpg" width=200  alt="accessibility text">
  <span> <img align="right" src="https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-shape-predictor-human-human-mappring/Results/snow.png"> </span>
</div>


