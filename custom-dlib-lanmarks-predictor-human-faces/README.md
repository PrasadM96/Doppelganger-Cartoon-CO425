
Train a model to predict the 35 landmarks on human faces<a name="TOP"></a>
===================

## Training the model ##

  Shape predictors, also called landmark predictors, are used to predict key (x, y)-coordinates of a given “shape”. dlib’s facial landmark predictor is used to training our custom dataset. Dlib is the most common, well-known shape predictor which is used to localize individual facial structures, including the:

  * Eyes
  * Eyesbrows
  * Nose
  * Mouth / Lips
  * Jawline

  To estimate the landmark locations, the algorithm:

  * Examines a sparse set of input pixel intensities (i.e., the “features” to the input model)
  * Passes the features into an Ensemble of Regression Trees (ERT)
  * Refines the predicted locations to improve accuracy through a cascade of regressors
  
  Here, we used this dlib shape predictor to train our custome dataset.
  
  
___Dataset___
  : iBUG 300-W dataset (https://ibug.doc.ic.ac.uk/resources/300-W/) which has is used to train the model. To create the iBUG-300W dataset, researchers manually and painstakingly annotated and labeled each of the 68 coordinates on a total of 7,764 images.
 
___Steps___

  * [parse_xml.py](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-human-faces/parse_xml.py)
    * iBUG 300-W dataset has images with 68 landmarks. In this case, only 35 landmarks are considered. [labels_ibug_300W_train.xml](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-human-faces/labels_ibug_300W_train.xml) as the input file and [reduced_labels_ibug_300W_train_35_points.xml](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-human-faces/reduced_labels_ibug_300W_train_35_points.xml) as the output file have to given as command line arguments.
    
  * [train_shape_predictor.py](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-human-faces/train_shape_predictor.py)
    * In this program, the [reduced_labels_ibug_300W_train_35_points.xml](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-human-faces/reduced_labels_ibug_300W_train_35_points.xml) is given as xml_path and it creates the predictor_points.dat file which is used to predict the landmarks of test images.
   
   ## Testing the model ##
   
   
___Steps___

  * [predict_points.py](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-human-faces/predict_points.py)
    * In this program, some human images in the test_images folder is given as inputs and get the predicted landmarks for that images. The resulted images are saved in the results folder.
    
  * [evaluate_shape_predictor.py](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-human-faces/evaluate_shape_predictor.py)
    * In this program, the [reduced_labels_ibug_300W_test_35_points.xml](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-human-faces/reduced_labels_ibug_300W_test_35_points.xml) is given as xml_path and the predictor_points.dat as model and it outputs an error value after evaluating the results for the images which is in reduced_labels_ibug_300W_test_35_points.xml
    
 ## Structure ##

### File Description ###

  * [parse_xml.py](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-human-faces/parse_xml.py) : to reduce the xmlfile from 68 points to 35 points
  * [train_shape_predictor.py](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-human-faces/train_shape_predictor.py) : train the model for predicting 35 lanmarks on human face
  * [predict_points.py](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-human-faces/predict_points.py) : test the model for some custom images
  * [evaluate_shape_predictor.py](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-human-faces/evaluate_shape_predictor.py) :  to evaluate the model for training set and test set and gives the MAE as output
  * [labels_ibug_300W.xml](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-human-faces/labels_ibug_300W.xml)
  * [labels_ibug_300W_test.xml](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-human-faces/labels_ibug_300W_test.xml) : xml file for test with 68 landmarks
  * [labels_ibug_300W_train.xml](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-human-faces/labels_ibug_300W_train.xml) : xml file for train with 68 landmarks
  * [reduced_labels_ibug_300W_test_35_points.xml](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-human-faces/reduced_labels_ibug_300W_test_35_points.xml) : xml file for test with 35 landmarks
  * [reduced_labels_ibug_300W_train_35_points.xml](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-human-faces/reduced_labels_ibug_300W_train_35_points.xml) : xml file for train with 35 landmarks
  

### Folder Description ###

  * [test_images](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/tree/main/custom-dlib-lanmarks-predictor-human-faces/test_images) : containing some custom test images
  * [results](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/tree/main/custom-dlib-lanmarks-predictor-human-faces/results) : results of the custom images
  
  
  ## Results ##
 ![picture alt](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-human-faces/results/Alfredo_Linguini_Real.png "Title is optional")
 ![picture alt](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-human-faces/results/Alice_real1.png "Title is optional")
 ![picture alt](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-human-faces/results/Dora_real.png "Title is optional")
 ![picture alt](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-human-faces/results/Flynn_rider_real2.png "Title is optional")
 ![picture alt](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-human-faces/results/Flynn_rider_real3.png "Title is optional")
 ![picture alt](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-human-faces/results/Hiro_real2.png "Title is optional")
 ![picture alt](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-human-faces/results/Homer_simpson_real1.png "Title is optional")
 ![picture alt](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-human-faces/results/Homer_simpson_real3.png "Title is optional")
 ![picture alt](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-human-faces/results/Mr_potato_head_real1.png "Title is optiona")
 ![picture alt](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-human-faces/results/Nanny_pelakai_real1.png "Title is optional")
 ![picture alt](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-human-faces/results/Prince_hans_real.png "Title is optional")
 ![picture alt](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-human-faces/results/Roxanne_real1.png "Title is optional")
 ![picture alt](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-human-faces/results/Sid_real1.png "Title is optional")
 ![picture alt](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-human-faces/results/baby_and_father.png "Title is optional")
 ![picture alt](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-human-faces/results/baby_and_father2.png "Title is optional")
 ![picture alt](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-human-faces/results/black_widow_real1.png "Title is optional")
 ![picture alt](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-human-faces/results/edna_real3.png "Title is optiona")
 ![picture alt](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-human-faces/results/elsa_real4.png "Title is optional")
 ![picture alt](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-human-faces/results/fawn_real1_1.png "Title is optional")
 ![picture alt](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-human-faces/results/gothel.png "Title is optional")
 ![picture alt](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-human-faces/results/jack_forest_real5.png "Title is optional")
 ![picture alt](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-human-faces/results/jesmine_Real.png "Title is optional")
 ![picture alt](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-human-faces/results/merida_real.png "Title is optional")
 ![picture alt](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-human-faces/results/percy_real.png "Title is optional")
 ![picture alt](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-human-faces/results/pocahontas_real1.png "Title is optional")
 ![picture alt](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-human-faces/results/princess_ariel_real.png "Title is optional")
 ![picture alt](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-human-faces/results/russel.png "Title is optional")
 ![picture alt](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-human-faces/results/snow_white_real.png "Title is optional")






  


