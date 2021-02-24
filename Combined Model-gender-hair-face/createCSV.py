import face.modules as modules
from face.metaData import img_folder_path, cartoon_predictor, cartoon_expected_dis, cartoon_expected_areas
from hair import evaluate
import numpy as np


# def main():
#     img_path_list = modules.getImagePaths(img_folder_path)

#     csv_list = []

#     for filepath in img_path_list:
#         shape_points = modules.predictPoints(filepath, cartoon_predictor)
#         print(filepath)
#         if(len(shape_points) != 0):
#             hair_output = evaluate.getHairVector(filepath)
#             result = modules.makeVector(
#                 shape_points, cartoon_expected_dis, cartoon_expected_areas)

#             if len(result) > 0:
#                 tmp = []
#                 tmp.append(result)
#                 tmp.append(hair_output)
#                 tmp.append(filepath)
#                 csv_list.append(tmp)

#     modules.makeCsv(csv_list)


def extractPointsHair(batchPaths):
    batchImgPaths = []
    batchImgFaceFeatures = []
    batchImgHair = []

    for imagePath in batchPaths:
        shape_points = modules.predictPoints(imagePath, cartoon_predictor)
        if(len(shape_points) != 0):
            faceFeatures = modules.makeVector(
                shape_points, cartoon_expected_dis, cartoon_expected_areas)
            hair_output = evaluate.getHairOutput(imagePath)
            batchImgPaths.append(imagePath)
            batchImgHair.append(hair_output)
            batchImgFaceFeatures.append(faceFeatures)

    return batchImgPaths, batchImgHair, batchImgFaceFeatures


def main2():
    img_path_list = modules.getImagePaths(img_folder_path)

    csv_list = []
    i = 608
    # for (b, i) in enumerate(range(0, len(img_path_list), 32)):
    batchPaths = img_path_list[i:i + 32]
    batchImgPaths, batchImgHair, batchImgFaceFeatures = extractPointsHair(
        batchPaths)

    batchImgHair = np.vstack(batchImgHair)

    hairFeatures = evaluate.getHairVector(batchImgHair)
    for i in range(len(batchImgPaths)):
        # print(batchImgPaths[0])
        # print(type(list(hairFeatures[0])))
        # print(type(batchImgFaceFeatures[0]))
        temp = []
        temp.append(batchImgFaceFeatures[i])
        temp.append(list(hairFeatures[i]))
        temp.append(batchImgPaths[i])
        csv_list.append(temp)

    modules.makeCsv(csv_list)


if __name__ == '__main__':
    # main2()
    img_path_list = modules.getImagePaths(img_folder_path)
    print(len(img_path_list))
    # model_output = evaluate.getHairVector('hair\img2\OIPGV1X6ZGR.jpg') * 10000
    # model_output1 = evaluate.getHairVector(
    #     'hair\img2\OIPGV1X6ZGR - Copy.jpg') * 10000

    # print(evaluate.findEuclideanDistance(model_output[0], model_output1[0]))
