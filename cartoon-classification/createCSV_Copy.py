
import csv
import face.modules as modules
from face.metaData import img_folder_path, cartoon_predictor, cartoon_expected_dis, cartoon_expected_areas
from hair import evaluate
print('sfs')


# # %%

# img_path_list = modules.getImagePaths(img_folder_path)

# csv_list = []

# # %%
# imagePaths, shapeVectorList = modules.predictPoints2(
# img_path_list, cartoon_predictor)
# # %%
# mask_output_list = evaluate.getHairVector3(imagePaths)
# # %%
# mask_vgg_output = evaluate.getVGGOuptput(mask_output_list)


def main():

    img_path_list = modules.getImagePaths(img_folder_path)

    csv_list = []

    imagePaths, shapeVectorList = modules.predictPoints2(
        img_path_list, cartoon_predictor)
    mask_output_list = evaluate.getHairVector3(imagePaths)
    mask_vgg_output = evaluate.getVGGOuptput(mask_output_list)

    for imagePath, i in enumerate(imagePaths):
        tmp = []
        tmp.append(shapeVectorList[i])
        tmp.append(mask_vgg_output[i])
        tmp.append(imagePath)
        csv_list.append(tmp)

    modules.makeCsv(csv_list)


if __name__ == '__main__':
    main()

    # model_output = evaluate.getHairVector('hair\img2\OIPGV1X6ZGR.jpg') * 10000
    # model_output1 = evaluate.getHairVector(
    #     'hair\img2\OIPGV1X6ZGR - Copy.jpg') * 10000

    # print(evaluate.findEuclideanDistance(model_output[0], model_output1[0]))
