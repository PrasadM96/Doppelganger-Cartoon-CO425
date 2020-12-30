Train a model to predict the 35 landmarks on cartoon faces<a name="TOP"></a>
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
  : Custom dataset created by us including 77 different characters of __Disney Cartoons__ . The dataset is containing 400 different images and most of cartoon characters have more than 5 images. All the number of images which is used in this task is 800 images including the mirror image of each 400 images.
 
___Steps___
    
  * [train_shape_predictor.py](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-cartoon-faces/train_shape_predictor.py)
    * In this program, the [Both_Cartoon_faces_with_landmakrs_train.xml](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-cartoon-faces/Both_Cartoon_faces_with_landmakrs_train.xml) is given as xml_path and it creates the predictor_points.dat file which is used to predict the landmarks of test images.
   
   ## Testing the model ##
    
___Steps___

  * [predict_points.py](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-cartoon-faces/predict_points.py)
    * In this program, some cartoon images in the test_images folder is given as inputs and get the predicted landmarks for that images. The resulted images are saved in the results folder. Here We used the [mmod_human_face_detector.dat](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-cartoon-faces/mmod_human_face_detector.dat) in the program for detect the face of cartoon iamges.

    
 ## Structure ##

### File Description ###

  * [train_shape_predictor.py](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-cartoon-faces/train_shape_predictor.py) : train the model for predicting 35 lanmarks on cartoon face
  * [predict_points.py](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-cartoon-faces/predict_points.py) : test the model for some custom images
  *[Both_Cartoon_faces_with_landmakrs_train.xml](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-cartoon-faces/Both_Cartoon_faces_with_landmakrs_train.xml) : xml file for train with 35 landmarks of cartoon images 
  * [Both_Cartoon_face_with_landmarks_test.xml](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-cartoon-faces/Both_Cartoon_face_with_landmarks_test.xml) : xml file for test with 35 landmarks of cartoon images 
  * [mmod_human_face_detector.dat](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-cartoon-faces/mmod_human_face_detector.dat) : dat file for detect the face of cartoon images.

### Folder Description ###

  * [Test_images](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/tree/main/custom-dlib-lanmarks-predictor-cartoon-faces/Test_images) : containing some custom test images
  * [Results](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/tree/main/custom-dlib-lanmarks-predictor-cartoon-faces/Results) : results of the custom images
  
  
  ## Results ##
 ![picture alt](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-cartoon-faces/Results/Aunt_cass.png "Title is optional")
 ![picture alt](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-cartoon-faces/Results/Elena2.png "Title is optional")
 ![picture alt](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-cartoon-faces/Results/Hiro_1.png "Title is optional")
 ![picture alt](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-cartoon-faces/Results/Honey_Lemon.png "Title is optional")
 ![picture alt](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-cartoon-faces/Results/Honey_Lemon2.png "Title is optional")
 ![picture alt](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-cartoon-faces/Results/Miguel_Rivera1.png "Title is optional")
 ![picture alt](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-cartoon-faces/Results/Miguel_Rivera13.png "Title is optional")
 ![picture alt](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-cartoon-faces/Results/Miguel_Rivera2.png "Title is optional")
 ![picture alt](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-cartoon-faces/Results/Moana.png "Title is optiona")
 ![picture alt](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-cartoon-faces/Results/Tumblur.png "Title is optional")
 ![picture alt](https://github.com/PrasadM96/Doppelganger-Cartoon-CO425/blob/main/custom-dlib-lanmarks-predictor-cartoon-faces/Results/test1.png "Title is optional")
 



