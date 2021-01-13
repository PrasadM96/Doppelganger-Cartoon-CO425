
from PIL import Image
import numpy as np
#import face_recognition
import cv2
import os
import json
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16   
import dlib 
import imutils
from keras.models import Model
from pickle import dump
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
import csv
from mtcnn import MTCNN
import pandas
import ast 
from operator import itemgetter

"""#  ***Display image***"""

image_formats = ["png", "jpg","jpeg"]; # Let suppose we want to display png & jpg images (specify more if you want)

def show_images(image_file_name):
    print("Displaying ", image_file_name)
    img=mpimg.imread(image_file_name)
    imgplot = plt.imshow(img)
    plt.show()

"""#  ***Get all image paths from image directory***"""

def get_image_paths(current_dir):
    files = os.listdir(current_dir);
    paths = []; # To store relative paths of all png and jpg images

    for file in files:
        file = file.strip()
        for image_format in image_formats:
            image_paths = glob.glob(os.path.join( current_dir,file,"*." + image_format))
            if image_paths:
                paths.extend(image_paths);

    return paths

"""#  ***Finding Distance Matrices  ?????***"""

def findManhattanDistance(source_representation, test_representation):
    return np.linalg.norm(source_representation - test_representation, ord=1)
 
def findEuclideanDistance(source_representation, test_representation):
    return np.linalg.norm(source_representation - test_representation)

"""#  ***Verifying faces depending on distances***"""

def verifyFace(img1, img2):
    manhattan_distance = findManhattanDistance(img1, img2)
    euclidean_distance = findEuclideanDistance(img1, img2)
    print("Manhattan distance : {}".format(manhattan_distance))
    print("Euclidean distance : {}".format(euclidean_distance))

"""#  ***Writing features to a CSV file***"""

def make_csv(temp):
    with open(r'G:\Education\Semester 8\425-FYP\Sem 7\data3.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(temp)

"""#  ***Pre-process images batchwise***"""

detector = dlib.cnn_face_detection_model_v1(r"G:\Education\Semester 8\425-FYP\Sem 7\mmod_human_face_detector.dat")

def pre_process_batch(batchPaths):
   batchImages = []
   batchImgPaths=[]
   for imagePath in batchPaths:
          # load the input image using the Keras helper utility
          # while ensuring the image is resized to 224x224 pixels
          #image = load_img(imagePath)
          #image = np.array(image, dtype = float)
          #image = img_to_array(image)
          # preprocess the image by (1) expanding, the dimensions and
          # (2) subtracting the mean RGB  pixel intensity from the
          # ImageNet dataset
          image = cv2.imread(imagePath)
          #image = imutils.resize(image,width=224)
          faces_cnn = detector(image)
          
          a=0
          print(imagePath)
          cord = [face.rect for face in faces_cnn]
          if(len(cord)>0):
              for face in faces_cnn:
                   print(face.rect,imagePath)
                   x = face.rect.left()-a
                   y = face.rect.top()-a
                   w = face.rect.right() - x + a
                   h = face.rect.bottom() - y + a
                   image = image[y:y+h, x:x+w]
                 
              image = cv2.resize(image, (224, 224))
              image = img_to_array(image)
              image = np.expand_dims(image, axis=0)
              image /= 255.0
          # image -= 1
          # image = preprocess_input(image)
          # add the image to the batch
        
              batchImages.append(image)
              batchImgPaths.append(imagePath)
        
   return batchImages,batchImgPaths

"""#***Pre-process single image***"""

def pre_process_single(detector,image):
  face_locations = detector[0]['box']
  a=15
  image_copy = image.copy()
  x, y, w, h = face_locations
  image = image[y-a:y+h+a, x-a:x+w+a]
  image = cv2.resize(image, (224, 224))
  cv2.rectangle(image_copy,  (x-a,y-a),   (x+w+a,y+h+a), (0, 255, 0), 8)
  plt.axis('off')
  plt.imshow(image_copy)
  image = img_to_array(image)
  image = np.expand_dims(image, axis=0)
  # image = preprocess_input(image)
  image /= 255.0
  # image -= 1

  return image

"""# ***Extract features from human images***"""

def extract_features_human(imagePath):
  image = load_img(imagePath)
  image = np.array(image)
  # image = image.astype(np.uint8)
  detector = MTCNN()
  detector=detector.detect_faces(image)
  print(detector)
  if(len(detector)!=0):
    image = pre_process_single(detector,image)
    real_features = model.predict(image)
    return real_features
  else:
    print('No face detected.Please try with another image!!')
    return []


"""# ***Extract features from cartoon images***"""

model = VGG16(weights=r'G:\Education\Semester 8\425-FYP\Dlib landmarks git\custom-dlib-shape-predictor-cartoon-human-mapping\NEW IMAGESET\Cartoon_face_with_landmarks')
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
def extract_features_cartoon(directory_name):
    image_paths = get_image_paths(directory_name)
    tempList =[]
    for (b, i) in enumerate(range(0, len(image_paths), 32)):
        batchPaths = image_paths[i:i + 32]
        batchImages,batchImgPaths = pre_process_batch(batchPaths)
        batchImages = np.vstack(batchImages)
        
        features = model.predict(batchImages, batch_size=32) 
        for i in range(len(batchImgPaths)):
          temp=[]
          temp.append(list(features[i]))
          temp.append(batchImgPaths[i])
          tempList.append(temp)
         

    return tempList


data=extract_features_cartoon(r'G:\Education\Semester 8\425-FYP\Sem 7\Cartoon_face_with_landmarks')

make_csv(data)



img_path = r'G:\Education\Semester 8\425-FYP\Sem 7\alice.jpg';

real_features = extract_features_human(img_path)

csv_results = pandas.read_csv(r'G:\Education\Semester 8\425-FYP\Sem 7\data3.csv',header=None)

euclidean_dis_list = []

if len(real_features)!=0:
  for index,result in enumerate(csv_results[0]):
    # print(result)
    # print(index)
    res = ast.literal_eval(result) 
    res = np.asarray(res)
    # verifyFace(real_features[0],res)
    euclidean_distance = findEuclideanDistance(real_features[0],res)
    euclidean_dis_list.append(euclidean_distance)

  indices, sorted_euclidean_dis = zip(*sorted(enumerate(euclidean_dis_list), key=itemgetter(1)))
  
  fig =plt.figure(1)
  img=load_img(img_path)
  img = np.array(img)
  plt.subplot(1,6,1)
  plt.axis('off')
  plt.title('real', fontsize=8)
  plt.imshow(img)
  
  
  for i in range(5):
     count=2
     print(sorted_euclidean_dis[i])
     resulted_image = load_img(csv_results[1][indices[i]])
     resulted_image = np.array(resulted_image)
     plt.subplot(1,6,count)
     plt.axis('off')
     plt.title(i+1, fontsize=8)
     plt.imshow(resulted_image)
     count = count+1
  plt.show()
#fig.savefig(r"G:/Education/Semester 8/425-FYP/Dlib landmarks git/custom-dlib-shape-predictor-human-human-mappring/eric_match_1.png", bbox_inches='tight', dpi=150)
  plt.close()
'''
for i,name in enumerate(results[1]):
  print(i,name)
'''
# with open('/content/data3.csv','rt')as f:
#   data = csv.reader(f)
#   for row in data:
#         print(row[0])

"""**Problem**"""

