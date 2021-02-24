import pandas
import ast
import numpy as np
from operator import itemgetter
from keras.preprocessing.image import load_img
from matplotlib import pyplot as plt
import face.modules as md
from face.metaData import human_predictor, human_expected_areas, human_expected_dis
import hair.evaluate as evaluate
from gender.malFemalDetector import age_gender_detector

csv_results_set = pandas.read_csv(
    r'data7_new.csv', header=None)


def main():
    euclidean_dis_list = []
    w = 0.8
    imgPath = r'cartoon_images\anna1.jpeg'
    shape_points = md.predictPoints(imgPath, human_predictor)
    face_vector = md.makeVector(shape_points, human_expected_dis,
                                human_expected_areas)

    mask_output = evaluate.getHairOutput(imgPath)
    hair_vector = evaluate.getHairVector(mask_output)

    gender_output = age_gender_detector(imgPath)

    # for index, result in enumerate(csv_results[0]):
    #     result = ast.literal_eval(result)
    #     break
    print(gender_output.lower())

    m = 0
    print(type(csv_results_set))
    print(csv_results_set.shape)
    csv_results = csv_results_set[csv_results_set[3] == gender_output.lower()]
    # print(filtered_csv[3])
    print(csv_results.shape)
    # print(type(csv_results))
    # print(csv_results)

    for face, hair in zip(csv_results[0], csv_results[1]):
        faceVal = list(ast.literal_eval(face))
        hairVal = list(ast.literal_eval(hair))

        euclidean_distance_face = md.findEuclideanDistance(
            np.array(face_vector, dtype=float), np.array(faceVal, dtype=float))
        euclidean_distance_hair = md.findEuclideanDistance(
            np.array(hair_vector, dtype=float), np.array(hairVal, dtype=float)) * 10000
        euclidean_distance = w * euclidean_distance_face + \
            (1-w) * euclidean_distance_hair
        euclidean_dis_list.append(euclidean_distance)
        # print(m, euclidean_distance_face, euclidean_distance_hair)
        m = m+1
    indices, sorted_euclidean_dis = zip(
        *sorted(enumerate(euclidean_dis_list), key=itemgetter(1)))

    k = 0
    csv_results = list(csv_results[2])
    for j in indices:

        print(k, csv_results[j], j)
        k = k+1

    fig = plt.figure(1)
    print(indices)

    j = 0
    for i in range(33, 39):
        # print(csv_results[2][indices[i]])
        print(indices[i])
        print(sorted_euclidean_dis[i], indices[i])
        print(csv_results[indices[i]])
        resulted_image = load_img(csv_results[indices[i]])
        resulted_image = np.array(resulted_image)

        plt.subplot(1, 6, j+1)
        plt.axis('off')
        j = j+1
        plt.title(i+1, fontsize=8)
        plt.imshow(resulted_image)

    plt.show()
    fig.savefig(r"after\jackfrost_w08.png", dpi=190)
    plt.close()


if __name__ == '__main__':
    main()
