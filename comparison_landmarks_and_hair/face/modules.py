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
from face.metaData import img_folder_path, cartoon_predictor, cartoon_expected_dis, cartoon_expected_areas
from face.metaData import image_formats, detector, frontal_face_detector


def calculateDis(point1, point2):
    sqauredDis = pow(point1[0]-point2[0], 2) + pow(point1[1]-point2[1], 2)
    return math.sqrt(sqauredDis)


def findEuclideanDistance(source_representation, test_representation):
    return np.linalg.norm(source_representation - test_representation)


def makeVector(shapeArr, dis_points, area_points):
    res = []

    if len(shapeArr) > 0:
        for i, pno in enumerate(dis_points):
            dis = calculateDis(shapeArr[pno[0]], shapeArr[pno[1]])
            res.append(dis)
            dis = 0

        for j, pno in enumerate(area_points):
            area = polygonArea(
                [shapeArr[pno[0]], shapeArr[pno[1]], shapeArr[pno[2]]])
            res.append(area)
    return res


def makeCsv(temp):
    with open(r'face\data7.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(temp)


def getImagePaths(current_dir):
    files = os.listdir(current_dir)
    paths = []  # To store relative paths of all png and jpg images

    for file in files:
        file = file.strip()
        for image_format in image_formats:
            image_paths = glob.glob(os.path.join(
                current_dir, file, "*." + image_format))
            if image_paths:
                paths.extend(image_paths)

    return paths


def predictPoints(filename, prd):
    print(filename)
    img = cv2.imread(filename)
    img = imutils.resize(img, width=224)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fontal_face = True
    faces_cnn = frontal_face_detector(gray)
    s = [face for face in faces_cnn]
    if(len(s) == 0):
        faces_cnn = detector(gray)
        fontal_face = False
    shape = []

    for face in faces_cnn:
        if fontal_face:
            (x, y, w, h) = face_utils.rect_to_bb(face)
            shape = prd(gray, face)
        else:
            x = face.rect.left()
            y = face.rect.top()
            w = face.rect.right() - x
            h = face.rect.bottom() - y
        # cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 1)
            shape = prd(gray, face.rect)

        shape = face_utils.shape_to_np(shape)

        # plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
        # plt.scatter(shape[:, 0], shape[:, 1], c='r', s=1)
        # plt.show()

    return shape


def predictPoints2(image_list, prd):
    imagePaths = []
    shapeList = []
    for filename in image_list:
        img = cv2.imread(filename)
        img = imutils.resize(img, width=224)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fontal_face = True
        faces_cnn = frontal_face_detector(gray)
        s = [face for face in faces_cnn]
        if(len(s) == 0):
            faces_cnn = detector(gray)
            fontal_face = False
        shape = []

        for face in faces_cnn:
            if fontal_face:
                (x, y, w, h) = face_utils.rect_to_bb(face)
                shape = prd(gray, face)
            else:
                x = face.rect.left()
                y = face.rect.top()
                w = face.rect.right() - x
                h = face.rect.bottom() - y
            # cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 1)
                shape = prd(gray, face.rect)

            shape = face_utils.shape_to_np(shape)
            if(len(shape) != 0):
                result = makeVector(
                    shape, cartoon_expected_dis, cartoon_expected_areas)
                imagePaths.append(filename)
                shapeList.append(result)

            # plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
            # plt.scatter(shape[:, 0], shape[:, 1], c='r', s=1)
            # plt.show()

    return imagePaths, shapeList


def polygonArea(vertices):
    # A function to apply the Shoelace algorithm
    numberOfVertices = len(vertices)
    sum1 = 0
    sum2 = 0

    for i in range(0, numberOfVertices-1):
        sum1 = sum1 + vertices[i][0] * vertices[i+1][1]
        sum2 = sum2 + vertices[i][1] * vertices[i+1][0]

    # Add xn.y1
    sum1 = sum1 + vertices[numberOfVertices-1][0]*vertices[0][1]
    # Add x1.yn
    sum2 = sum2 + vertices[0][0]*vertices[numberOfVertices-1][1]

    area = abs(sum1 - sum2) / 2

    return area
