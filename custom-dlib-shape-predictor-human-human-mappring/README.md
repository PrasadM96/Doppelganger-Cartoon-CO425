# Human - Human Mapping #

### Training for human face landmarks detection ###

  * [custom-dlib-lanmarks-predictor-human-faces](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/tree/main/custom-dlib-lanmarks-predictor-human-faces)
    * This folder is containing the python files and results of the training a model for human landmarks detection
    
    
 ### Method Explanantion ###
  * In this method, the comparison is done between human images.
  * _Dataset_
    * The data set is containing the cartoon images and the already known Doppelganger human images for the cartoon characters (70 characters, 226 images) in folders which are 
    named using cartoon character's name.
  * _Steps_
    * Detect the landmarks of the faces of human images using the model
    * Calculate the distnaces and areas btewenn landmarks
    * Save that distances and areas in a CSV file along with the path for the corresponding cartoon images 
    * Insert an unknown images to find the matching cartoon character
    * Detect the landmarks, Calculate the distances and areas between landmarks and Get the feature vector of the unknown image
    * Compare that feature vector with the saved feature vectors in CSV file using _Euclidean distance_
    * Display the 5 top matching cartoon images for the given image
  * Results for some test  images are displayed below.
      
## Results ##

<div style="height:300">
<img src="https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-shape-predictor-human-human-mappring/Test%20images/alice.jpg" width=200 alt="accessibility text">
 <span> <img align="right" src="https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-shape-predictor-human-human-mappring/Results/alice_match.png"> </span>
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

