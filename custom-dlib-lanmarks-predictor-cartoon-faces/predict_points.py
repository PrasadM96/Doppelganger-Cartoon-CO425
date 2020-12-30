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



    # return shape

###############################################################################
image_formats = ["png", "jpg","jpeg"];
# construct the argument parser and parse the arguments
model_path = "predictor_points.dat"
foldername = r"Test_images"



#
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model_path)

#shape_points = predictPoints('Test_images/cartoon1.jpeg')

for filename in os.listdir(foldername):
       filename=os.path.join(foldername,filename)
       predictPoints(filename)
