
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
  : iBUG 300-W dataset which has is used to train the model. To create the iBUG-300W dataset, researchers manually and painstakingly annotated and labeled each of the 68 coordinates on a total of 7,764 images.
 
___Steps___

  * parse_xml.py
    * iBUG 300-W dataset has images with 68 landmarks. In this case, only 35 landmarks are considered. labels_ibug_300W_train.xml as the input file and reduced_labels_ibug_300W_train_35_points.xml as the output file have to given as command line arguments.
    
  * train_shape_predictor.py
    * In this program, the reduced_labels_ibug_300W_train_35_points.xml is given as xml_path and it creates the predictor_points.dat file which is used to predict the landmarks of test images.
   
   ## Testing the model ##
   
   
___Steps___

  * predict_points.py
    * In this program, some human images in the test_images folder is given as inputs and get the predicted landmarks for that images. The resulted images are saved in the results folder.
    
  * evaluate_shape_predictor.py
    * In this program, the reduced_labels_ibug_300W_test_35_points.xml is given as xml_path and the predictor_points.dat as model and it outputs an error value after evaluating the results for the images which is in reduced_labels_ibug_300W_test_35_points.xml
    
 ## Structure ##

### File Description ###

  * parse_xml.py : to reduce the xmlfile from 68 points to 35 points
  * train_shape_predictor.py : train the model for predicting 35 lanmarks on human face
  * predict_points.py : test the model for some custom images
  * evaluate_shape_predictor.py :  to evaluate the model for training set and test set and gives the MAE as output
  * labels_ibug_300W.xml
  * labels_ibug_300W_test.xml : xml file for test with 68 landmarks
  * labels_ibug_300W_train.xml : xml file for train with 68 landmarks
  * reduced_labels_ibug_300W_test_35_points.xml : xml file for test with 35 landmarks
  * reduced_labels_ibug_300W_train_35_points.xml : xml file for train with 35 landmarks
  

### Folder Description ###

  * test_images : containing some custom test images
  * results : results of the custom images




  


