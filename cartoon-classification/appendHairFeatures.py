from csv import writer
from csv import reader
import pandas
import ast
import numpy as np
from operator import itemgetter
from keras.preprocessing.image import load_img
from matplotlib import pyplot as plt
import face.modules as md
from face.metaData import human_predictor, human_expected_areas, human_expected_dis
import hair.evaluate as evaluate


def updateCSv(inputCsv, outputCsv, newVals):
    with open(inputCsv, 'r') as read_obj,  open(outputCsv, 'w', newline='') as write_obj:
        csv_reader = reader(read_obj)
        csv_writer = writer(write_obj)
        i = 0
        for row in csv_reader:
            row.append(list(newVals[i]))
            csv_writer.writerow(row)
            i = i+1


def main():
    csv_results = pandas.read_csv(r'face\data6.csv', header=None)

    imagePaths = list(csv_results[1])
    hair_masks = evaluate.getHairVector3(imagePaths)
    print("hair mask done!")
    hair_vectors = evaluate.getVGGOuptput(hair_masks)
    print("hair vgg done!")
    updateCSv(r'face\data6.csv', r'face\data_updated.csv', hair_vectors)


if __name__ == '__main__':
    main()
