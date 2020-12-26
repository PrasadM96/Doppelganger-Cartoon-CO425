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
            print(area)
            res.append(area)
    return res

def makeCsv(temp):
    with open('data3.csv', 'w', newline='') as file:
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

def predictPoints(filename):
    img = cv2.imread(filename)
    img = imutils.resize(img,width=224)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray)
    shape=[]
    for rect in rects:
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 1)
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
        plt.scatter(shape[:,0], shape[:,1], c = 'r', s = 1)
        plt.show()
        
    return shape


def polygonArea(vertices):
  #A function to apply the Shoelace algorithm
  print(vertices)
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
model_path = "../datasets/ibug_300W_large_face_landmark_dataset/predictor_points.dat"
img_folder_path = r"G:\Education\Semester 8\425-FYP\custom-dlib-shape-predictor\drive-download-20201214T140143Z-001"
expected_dis=[[11,12],[12,13],[14,15],[15,16],[21,22],[23,24],[17,18],[18,21],[18,22],[18,23],[18,27],[18,19],[18,20],[25,29],[18,7],[18,5],[18,4],[18,6]]
expected_areas = [[11,12,13],[10,13,20]]
# initialize dlib's face detector (HOG-based) and then load our
# trained shape predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model_path)
###############################################################################



#111111111111111111111111111111111111111111

#csv list
###############################################################################
csv_list=[]

for folderename in os.listdir(img_folder_path):	
    foldername=os.path.join(img_folder_path,folderename)
    cartoon_img_path=[i for i in os.listdir(foldername) if "cartoon" in i]
    cartoon_img_path=os.path.join(img_folder_path,folderename,cartoon_img_path[0])
    
    
    for filename in os.listdir(foldername):
        filename=os.path.join(foldername,filename)
        print(filename)
        if ("cartoon" in filename) or ("Cartoon" in filename):
            continue
        else:
            shape_points = predictPoints(filename)    
            result=makeVector(shape_points,expected_dis,expected_areas)
            if len(result)>0:
                tmp=[]
                tmp.append(result)
                tmp.append(filename)
                tmp.append(cartoon_img_path)
                csv_list.append(tmp)    
    
makeCsv(csv_list)





csv_results = pandas.read_csv('data3.csv',header=None)

euclidean_dis_list = []

shape_points = predictPoints('real.jpg')    
realRes=makeVector(shape_points,expected_dis,expected_areas)


for index,result in enumerate(csv_results[0]):
    res = ast.literal_eval(result)
    

    euclidean_distance = findEuclideanDistance(np.array(realRes, dtype=float),np.array(res,dtype=float))
    euclidean_dis_list.append(euclidean_distance)

    indices, sorted_euclidean_dis = zip(*sorted(enumerate(euclidean_dis_list), key=itemgetter(1)))

    print(sorted_euclidean_dis)


for i in range(5):
    print(sorted_euclidean_dis[i])
    resulted_image = load_img(csv_results[2][indices[i]],target_size=(150,150))
    resulted_image = np.array(resulted_image)
    plt.figure()
    plt.axis('off')
    plt.imshow(resulted_image)







