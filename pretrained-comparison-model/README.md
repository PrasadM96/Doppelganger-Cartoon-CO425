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
<img src="https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/pretrained-comparison-model/Results/Gothel/gothen_real3.jpg"  width=200 alt="accessibility text">
 <span> <img align="right" src="https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/pretrained-comparison-model/Results/Gothel/gothen_match_2.png"> </span>
</div>
<hr>
<div style="height:300">
<img src="https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/pretrained-comparison-model/Results/Hiro/Hiro_real4.jpg" width=200  alt="accessibility text">
  <span> <img  align="right" src="https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/pretrained-comparison-model/Results/Hiro/Hiro_match_2.png"> </span>
</div>
<hr>
<div style="height:300">
<img src="https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/pretrained-comparison-model/Results/Merida/merida_real.jpg" width=200  alt="accessibility text">
  <span> <img align="right" src="https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/pretrained-comparison-model/Results/Merida/merida_match_1.png"> </span>
</div>
<hr>
<div style="height:300">
<img src="https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/pretrained-comparison-model/Results/Moona/Moana_real2.jpg" width=200  alt="accessibility text">
  <span> <img align="right" src="https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/pretrained-comparison-model/Results/Moona/Moana_match_1.png"> </span>
</div>
<hr>
<div style="height:300">
<img src="https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/pretrained-comparison-model/Results/Moona2/Moana_real3.jpg" width=200  alt="accessibility text">
  <span> <img align="right" src="https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/pretrained-comparison-model/Results/Moona2/Moana_match_1.png"> </span>
<hr>
<div style="height:300">
<img src="https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/pretrained-comparison-model/Results/Russel/russe.jpg" width=200  alt="accessibility text">
  <span> <img align="right" src="https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/pretrained-comparison-model/Results/Russel/russe_match_2.png"> </span>
</div>
<hr>
<div style="height:300">
<img src="https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/pretrained-comparison-model/Results/Woody/Woody.png" width=200  alt="accessibility text">
  <span> <img align="right" src="https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/pretrained-comparison-model/Results/Woody/woody_match_2.png"> </span>
</div> 
<hr>
<div style="height:300">
<img src="https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/pretrained-comparison-model/Results/jack_forest/jack_forest_real1.jpg" width=200  alt="accessibility text">
  <span> <img align="right" src="https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/pretrained-comparison-model/Results/jack_forest/jack_forest_match_2.png"> </span>
</div> 
<hr>
<div style="height:300">
<img src="https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/pretrained-comparison-model/Results/kristoff/kristoff_real.jpg" width=200  alt="accessibility text">
  <span> <img align="right" src="https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/pretrained-comparison-model/Results/kristoff/kristoff.png"> </span>
</div> 
<hr>
<div style="height:300">
<img src="https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/pretrained-comparison-model/Results/rapunzel/rapunzel_real5.jpg" width=200  alt="accessibility text">
  <span> <img align="right" src="https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/pretrained-comparison-model/Results/rapunzel/rapunzel_match.png"> </span>
</div>



