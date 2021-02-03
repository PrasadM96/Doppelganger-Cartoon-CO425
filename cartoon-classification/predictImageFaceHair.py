import pandas
import ast
import numpy as np
from operator import itemgetter
from keras.preprocessing.image import load_img
from matplotlib import pyplot as plt
import face.modules as md
from face.metaData import human_predictor, human_expected_areas, human_expected_dis
import hair.evaluate as evaluate


csv_results = pandas.read_csv(
    r'face\data7.csv', header=None)


def main():
    euclidean_dis_list = []
    w = 0.8
    imgPath = r'face\test_imgs\rapunzel.jpg'
    shape_points = md.predictPoints(imgPath, human_predictor)
    face_vector = md.makeVector(shape_points, human_expected_dis,
                                human_expected_areas)

    mask_output = evaluate.getHairOutput(imgPath)
    hair_vector = evaluate.getHairVector(mask_output)

    # for index, result in enumerate(csv_results[0]):
    #     result = ast.literal_eval(result)
    #     break
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
        print(euclidean_distance_face, euclidean_distance_hair)
        indices, sorted_euclidean_dis = zip(
            *sorted(enumerate(euclidean_dis_list), key=itemgetter(1)))

    fig = plt.figure(1)

    for i in range(5):
        print(sorted_euclidean_dis[i])
        resulted_image = load_img(csv_results[2][indices[i]])
        resulted_image = np.array(resulted_image)
        plt.subplot(1, 5, i+1)
        plt.axis('off')
        plt.title(i+1, fontsize=8)
        plt.imshow(resulted_image)

    plt.show()
    fig.savefig(r"face\rapunzel_match.png", dpi=150)
    plt.close()


if __name__ == '__main__':
    main()
