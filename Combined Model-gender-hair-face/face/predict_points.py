# USAGE
# python predict_eyes.py --shape-predictor eye_predictor.dat

# import the necessary packages

import argparse
from imutils.video import VideoStream
from imutils import face_utils
from matplotlib import pyplot as plt 
import imutils
import time
import dlib
import cv2
import math
import numpy as np
import matplotlib.image as mpimg
import csv
import os
import sys
import glob
import pandas
import ast 
from operator import itemgetter
from keras.preprocessing.image import load_img

def calculateDis(point1,point2):
    sqauredDis = pow(point1[0]-point2[0],2) + pow(point1[1]-point2[1],2) 
    return math.sqrt(sqauredDis)

def findEuclideanDistance(source_representation, test_representation):
    return np.linalg.norm(source_representation - test_representation)

def makeVector(shapeArr,dis_points,area_points):
    res=[]

    if len(shapeArr)>0:
        for i,pno in enumerate(dis_points):
            dis=calculateDis(shapeArr[pno[0]],shapeArr[pno[1]])
            res.append(dis)
            dis=0
            
        for j,pno in enumerate(area_points):
            area =polygonArea([shapeArr[pno[0]],shapeArr[pno[1]],shapeArr[pno[2]]])
            res.append(area)
    return res

def makeCsv(temp):
    with open('data6.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(temp)


def getImagePaths(current_dir):
    files = os.listdir(current_dir);
    paths = []; # To store relative paths of all png and jpg images

    for file in files:
        file = file.strip()
        for image_format in image_formats:
            image_paths = glob.glob(os.path.join( current_dir,file,"*." + image_format))
            if image_paths:
                paths.extend(image_paths);

    return paths

def predictPoints(filename, prd):
    img = cv2.imread(filename)
    img = imutils.resize(img,width=224)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces_cnn = detector(gray)
  
    shape=[]
    a=30
    
    for face in faces_cnn:
        x = face.rect.left()-a
        y = face.rect.top()-a
        w = face.rect.right() - x + a
        h = face.rect.bottom() - y + a

        cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 1)

        d=[(x,y),(w+x,w+y)]

        shape = prd(gray, face.rect)
        shape = face_utils.shape_to_np(shape)
        
        plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
        plt.scatter(shape[:,0], shape[:,1], c = 'r', s = 1)
        plt.show()
        
    return shape


def polygonArea(vertices):
  #A function to apply the Shoelace algorithm
  numberOfVertices = len(vertices)
  sum1 = 0
  sum2 = 0
  
  for i in range(0,numberOfVertices-1):
    sum1 = sum1 + vertices[i][0] *  vertices[i+1][1]
    sum2 = sum2 + vertices[i][1] *  vertices[i+1][0]
  
  #Add xn.y1
  sum1 = sum1 + vertices[numberOfVertices-1][0]*vertices[0][1]   
  #Add x1.yn
  sum2 = sum2 + vertices[0][0]*vertices[numberOfVertices-1][1]   
  
  area = abs(sum1 - sum2) / 2

  return area


###############################################################################      
image_formats = ["png", "jpg","jpeg"];
# construct the argument parser and parse the arguments
model_path = r"G:\Education\Semester 8\425-FYP\Dlib landmarks git\custom-dlib-shape-predictor-cartoon-human-mapping\Cartoon-front-face\cartoon_predictor_points_front_face.dat"
human_model_path = r"G:\Education\Semester 8\425-FYP\Dlib landmarks git\custom-dlib-shape-predictor-cartoon-human-mapping\Cartoon-front-face\predictor_points.dat"
img_folder_path = r"G:\Education\Semester 8\425-FYP\Dlib landmarks git\custom-dlib-shape-predictor-cartoon-human-mapping\Cartoon-front-face\Cartoon_face_with_landmarks"
#expected_dis=[[12,33],[12,31],[12,29],[12,27],[12,25],[12,24],[12,26],[12,28],[12,30],[12,32],[12,34],[12,6],[12,7],[12,10],[12,8],[12,9],[12,14],[12,15],[12,16],[12,17],[12,18],[0,2],[3,5],[6,7],[8,9],[14,18],[11,12],[12,13],[2,3],[7,8]]
#expected_human_distances =[[0,18],[1,18],[2,18],[3,18],[4,18],[5,18],[6,18],[7,18],[8,18],[9,18],[10,18],[21,18],[22,28],[17,18],[23,28],[24,28],[18,25],[18,26],[18,27],[18,28],[18,29],[11,13],[14,16],[21,22],[23,24],[25,29],[18,19],[18,20],[13,14],[22,23]]
#expected_areas = [[0,33,6,7,1,2],[2,7,10,3,8],[3,8,4,5,9,34],[31,33,29,11,10,7,6],[10,11,12,13],[10,8,9,34,32,30,31,12],[29,27,14,15,12,11],[12,13,17,18,28,30],[12,15,17,14,18],[]]
'''
expected_areas = [[0,33,6],[0,6,1],[6,7,1],[1,7,2],[1,7,10],[1,10,2],[2,10,8],[2,4,8],[8,4,9],[4,5,9],
                  [5,9,34],[31,33,6],[6,31,11],[6,7,11],[11,10,7],[10,11,12],[12,13,10],[10,13,8],[8,13,9],
                  [13,9,32],[9,32,34],[31,29,11],[29,14,11],[29,27,14],[11,14,15],[11,12,15],[12,15,16],
                  [12,16,17],[12,17,13],[13,17,18],[13,18,30],[13,18,30],[13,30,32],[14,15,16,17,18,19],
                  [14,21,22,23,18,20],[18,28,30],[18,28,26],[23,26,18],[24,23,26],[22,23,24],[21,22,24],
                  [21,25,24],[21,25,14],[27,14,25]]
'''
#expected_areas = [[0,6,33],[0,1,6],[1,6,7],[1,2,7],[2,7,10],[2,3,10],[3,8,10],[3,4,8],[4,8,9],[4,5,9],[5,9,34],[33,31,6],[31,6,11],[6,7,11],[7,10,11],[10,11,12],[10,12,13],[8,10,13],[8,9,13],[9,13,32],[9,32,34],[31,29,11],[13,30,32],[29,14,11],[11,14,15],[11,12,15],[15,16,12],[12,16,17],[12,17,13],[13,17,18],[13,18,30],[14,27,29],[14,15,16],[16,17,18],[18,28,30],[14,16,19],[16,19,18],[14,19,20],[18,19,20],[14,20,21],[20,21,22],[20,22,23],[18,20,23],[14,27,21],[18,23,28],[21,27,25],[21,22,25],[22,24,25],[22,24,26],[22,23,26],[23,28,26]]
#expected_human_areas =  [[0,11,21],[11,12,21],[12,22,21],[12,13,22],[13,22,17],[13,14,17],[14,17,23],[14,25,23],[23,15,24],[15,16,24],[10,16,24],[0,1,21],[1,19,21],[19,21,22],[17,19,22],[17,18,19],[17,18,20],[17,20,23],[20,23,24],[20,24,9],[24,9,10],[1,2,19],[20,9,8],[2,25,19],[26,25,19],[18,19,26],[18,26,27],[18,27,28],[18,20,28],[20,28,29],[20,29,8],[2,3,25],[25,26,27],[27,28,29],[29,8,7],[25,27,33],[27,29,33],[32,33,34],[33,34,29],[25,34,32],[31,32,34],[31,34,30],[30,34,29],[3,25,32],[30,29,7],[3,4,32],[4,31,32],[4,5,31],[5,6,31],[30,31,6],[30,6,7]]

human_expected_areas=[[0,11,21],[11,12,21],[12,21,22],[12,13,22],[13,17,22],[13,14,17],[17,14,23],[14,15,23],[15,23,14],[15,16,24],[16,24,10],[1,2,19],[1,0,19],[0,21,19],[21,22,19],[22,17,19],[17,19,18],[17,18,20],[17,23,20],[23,24,20],[20,24,10],[20,10,9],[20,9,8],[20,28,8],[18,20,28],[18,27,28],[18,27,26],[19,18,26],[2,19,26],[2,3,25],[25,26,2],[25,26,34],[26,27,34],[27,34,28],[28,29,34],[29,28,8],[29,8,7],[3,25,32],[25,33,32],[32,33,31],[33,31,30],[33,30,29],[30,29,7],[3,32,4],[4,5,32],[31,32,5],[31,30,5],[30,5,6],[30,6,7]]

cartoon_expected_areas=[[0,27,31],[0,1,31],[1,31,32],[1,12,32],[12,2,32],[12,23,2],[2,23,33],[23,29,33],[29,33,34],[29,30,34],[30,34,28],[25,22,3],[25,27,3],[27,31,3],[31,32,3],[32,2,3],[2,3,4],[2,4,5],[2,33,5],[33,34,5],[5,34,28],[5,28,26],[5,26,24],[5,9,24],[4,5,9],[4,8,9],[4,7,8],[3,7,4],[3,7,22],[22,20,6],[6,7,22],[6,7,11],[7,8,11],[8,11,9],[9,11,10],[9,10,24],[10,24,21],[20,6,14],[6,13,14],[14,13,15],[13,15,16],[13,16,10],[16,10,21],[20,14,18],[14,18,17],[15,18,17],[15,16,17],[16,17,19],[16,21,19]]

human_expected_dis = [[11,12],[12,13],[14,15],[15,16],[21,22],[23,24],[17,18],[17,18],[17,20],[18,19],[18,20],[18,11],[18,12],[18,13],
                      [18,21],[18,22],[18,14],[18,15],[18,16],[18,23],[18,24],[18,0],[18,1],[18,2],[18,3],[18,4],[18,5],[18,6],
                      [18,7],[18,8],[18,9],[18,10],[18,25],[18,26],[18,27],[18,28],[18,29],[18,34],[18,32],[18,31],[18,30],[18,33]]

cartoon_expected_dis = [[0,1],[1,12],[23,29],[29,30],[31,32],[33,34],[2,4],[2,3],[2,5],[4,3],[4,5],[4,0],[4,1],[4,12],[4,31],[4,32],[4,32],[4,29],[4,30],[4,33],[4,34],[4,27],[4,25],[4,22],[4,20],[4,18],[4,17],[4,19],[4,21],[4,24],[4,26],[4,28],[4,6],[4,7],[4,8],[4,9],[4,10],[4,11],[4,14],[4,15],[4,16],[4,13]]




# initialize dlib's face detector (HOG-based) and then load our
# trained shape predictor
print("[INFO] loading facial landmark predictor...")
#detector = dlib.get_frontal_face_detector()
# initialize cnn based face detector with the weights
detector = dlib.cnn_face_detection_model_v1(r"G:\Education\Semester 8\425-FYP\Dlib landmarks git\custom-dlib-shape-predictor-cartoon-human-mapping\Cartoon-front-face\mmod_human_face_detector.dat")
predictor = dlib.shape_predictor(model_path)
human_predictor = dlib.shape_predictor(human_model_path)
###############################################################################





img_path_list=getImagePaths(img_folder_path)


#111111111111111111111111111111111111111111

#csv list
###############################################################################

csv_list=[]

for filepath in img_path_list:	
    shape_points = predictPoints(filepath,predictor) 
    if(len(shape_points)!=0):
        result=makeVector(shape_points,cartoon_expected_dis,cartoon_expected_areas)
        
        if len(result)>0:
            tmp=[]
            tmp.append(result)
            tmp.append(filepath)
            csv_list.append(tmp)   
           
    
makeCsv(csv_list)



def predictPoints2(point,filename, prd):
    img = cv2.imread(filename)
    img = imutils.resize(img,width=224)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces_cnn = detector(gray)
  
    shape=[]
    a=30
    
    for face in faces_cnn:
        x = face.rect.left()-a
        y = face.rect.top()-a
        w = face.rect.right() - x + a
        h = face.rect.bottom() - y + a

        cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 1)

        d=[(x,y),(w+x,w+y)]

        shape = prd(gray, face.rect)
        shape = face_utils.shape_to_np(shape)
        
        plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
        plt.scatter(shape[point,0], shape[point,1], c = 'r', s = 1)
        plt.show()
        
    return shape





csv_results = pandas.read_csv(r'G:\Education\Semester 8\425-FYP\Dlib landmarks git\custom-dlib-shape-predictor-cartoon-human-mapping\Cartoon-front-face\data6.csv',header=None)


euclidean_dis_list = []

shape_points = predictPoints2(8,r'G:\Education\Semester 8\425-FYP\Dlib landmarks git\custom-dlib-shape-predictor-cartoon-human-mapping\Cartoon-front-face\flipped_Flynn Rider_1.png',predictor)    


realRes=makeVector(shape_points,human_expected_dis,human_expected_areas)



for index,result in enumerate(csv_results[0]):
    res = ast.literal_eval(result)
    

    euclidean_distance = findEuclideanDistance(np.array(realRes, dtype=float),np.array(res,dtype=float))
    euclidean_dis_list.append(euclidean_distance)

    indices, sorted_euclidean_dis = zip(*sorted(enumerate(euclidean_dis_list), key=itemgetter(1)))

fig =plt.figure(1)

for i in range(5):
    print(sorted_euclidean_dis[i])
    resulted_image = load_img(csv_results[1][indices[i]])
    resulted_image = np.array(resulted_image)
    plt.subplot(1,5,i+1)
    plt.axis('off')
    plt.title(i+1, fontsize=8)
    plt.imshow(resulted_image)

plt.show()
#fig.savefig(r"G:\Education\Semester 8\425-FYP\Dlib landmarks git\custom-dlib-shape-predictor-cartoon-human-mapping\Cartoon-front-face\silvermist_match.png", dpi=150)
plt.close()


