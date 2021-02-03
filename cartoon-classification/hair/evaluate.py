import os
import time
import numpy as np
import argparse
from keras.applications.vgg16 import VGG16
import cv2
import numpy as np
from keras.models import load_model
from keras.utils import CustomObjectScope
from scipy.misc import imread, imresize, imsave
import matplotlib.pyplot as plt
from hair.utils.custom_objects import custom_objects
from hair.utils.loss import np_dice_coef
from hair.nets.DeeplabV3plus import DeeplabV3plus
from hair.nets.Prisma import PrismaNet
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


model_1 = VGG16(
    weights=r'G:\Education\SEMESTER 7\CO421 FINAL YEAR PROJECT\PROJECT\vgg16_weights_tf_dim_ordering_tf_kernels.h5')
model_1 = Model(inputs=model_1.inputs, outputs=model_1.layers[-2].output)
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_path", default=r'hair\models\CelebA_PrismaNet_256_hair_seg_model.h5')

parser.add_argument('--input_shape', default=[256, 256])
args = parser.parse_args()


def blend_img_with_mask(img, alpha, img_shape):
    mask = alpha >= 0.99
    mask_3 = np.zeros(img_shape, dtype='float32')
    mask_3[:, :, 0] = 255
    mask_3[:, :, 0] *= alpha
    result = img*0.5 + mask_3*0.5
    return np.clip(np.uint8(result), 0, 255)


def evaluate(model_path, imgs_path, input_shape):
    with CustomObjectScope(custom_objects()):
        model = load_model(model_path)
       # model.summary()

    img = imread(imgs_path, mode='RGB')
    img_shape = img.shape
    input_data = img.astype('float32')
    input_data = imresize(img, input_shape)
    input_data = input_data / 255.
    input_data = (input_data - input_data.mean()) / input_data.std()
    input_data = np.expand_dims(input_data, axis=0)

    output = model.predict(input_data)

    mask = cv2.resize(output[0, :, :, 0], (img_shape[1],
                                           img_shape[0]), interpolation=cv2.INTER_LINEAR)

    mask3d = np.dstack([mask]*3)

    # plt.imshow(mask3d)
    # plt.show()

    return mask3d
    #img_with_mask = blend_img_with_mask(img, mask, img_shape)
    #imsave(r"G:\Education\Semester 8\425-FYP\Hair_Segmentation_Keras-master\Hair_Segmentation_Keras-master" + _, mask)


def preProcessImg(image):
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.0
    # image -= 1

    return image


def findEuclideanDistance(source_representation, test_representation):
    return np.linalg.norm(source_representation - test_representation)


def getHairOutput(imagePath):
    mask_output = evaluate(
        args.model_path, imagePath, args.input_shape)
    mask_output = preProcessImg(mask_output)

    return mask_output


def getHairVector(mask_output):
    # mask_output = evaluate(
    #     args.model_path, imagePath, args.input_shape)
    # image = preProcessImg(mask_output)
    # print(image.shape)
    model_output = model_1.predict(mask_output)

    return model_output


def getHairVector2(imagePath):
    mask_output = evaluate(
        args.model_path, imagePath, args.input_shape)
    image = preProcessImg(mask_output)
    # print(image.shape)
    # model_output = model_1.predict(image)

    return image


def getHairVector3(imagePaths):
    mask_output_list = []
    for imagePath in imagePaths:
        mask_output = evaluate(
            args.model_path, imagePath, args.input_shape)
        mask_output = preProcessImg(mask_output)
        mask_output_list.append(mask_output)
    return mask_output_list


def getVGGOuptput(mask_output_list):
    vgg_list = []
    for (b, i) in enumerate(range(0, len(mask_output_list), 32)):
        batch_masks = mask_output_list[i:i + 32]
        batchImgHair = np.vstack(batch_masks)
        model_output = model_1.predict(batchImgHair, batch_size=32)
        vgg_list.append(model_output)
    return vgg_list
# mask_output1 = evaluate(
#     args.model_path, r"img2\OIPGV1X6ZGR - Copy.jpg", args.input_shape)
# image1 = pre_process(mask_output1)
# print(image1.shape)

# model_output1 = model_1.predict(image1)
# print()

# model_output = model_output * 100
# model_output1 = model_output1 * 100

# print(findEuclideanDistance(model_output[0], model_output1[0]))


# https://github.com/Papich23691/Hair-Detection


# model_output = getHairVector('img2\OIPGV1X6ZGR.jpg')
# model_output1 = getHairVector('img2\OIPGV1X6ZGR - Copy.jpg')

# print(findEuclideanDistance(model_output[0], model_output1[0]))
