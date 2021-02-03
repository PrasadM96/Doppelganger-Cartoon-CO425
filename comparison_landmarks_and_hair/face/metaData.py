import dlib
import pandas

image_formats = ["png", "jpg", "jpeg"]
# G:\Education\Semester 8\425-FYP\Dlib landmarks git\custom-dlib-shape-predictor-cartoon-human-mapping\Cartoon-front-face\
model_path = r"face\cartoon_predictor_points_front_face.dat"
# G:\Education\Semester 8\425-FYP\Dlib landmarks git\custom-dlib-shape-predictor-cartoon-human-mapping\Cartoon-front-face\
human_model_path = r"face\predictor_points.dat"
# G:\Education\Semester 8\425-FYP\Dlib landmarks git\custom-dlib-shape-predictor-cartoon-human-mapping\Cartoon-front-face\
img_folder_path = r"face\Cartoon_face_with_landmarks"
# G:\Education\Semester 8\425-FYP\Dlib landmarks git\custom-dlib-shape-predictor-cartoon-human-mapping\Cartoon-front-face\



human_expected_areas = [[0, 11, 21], [11, 12, 21], [12, 21, 22], [12, 13, 22], [13, 17, 22], [13, 14, 17], [17, 14, 23], [14, 15, 23], [15, 23, 14], [15, 16, 24], [16, 24, 10], [1, 2, 19], [1, 0, 19], [0, 21, 19], [21, 22, 19], [22, 17, 19], [17, 19, 18], [17, 18, 20], [17, 23, 20], [23, 24, 20], [20, 24, 10], [20, 10, 9], [20, 9, 8], [
    20, 28, 8], [18, 20, 28], [18, 27, 28], [18, 27, 26], [19, 18, 26], [2, 19, 26], [2, 3, 25], [25, 26, 2], [25, 26, 34], [26, 27, 34], [27, 34, 28], [28, 29, 34], [29, 28, 8], [29, 8, 7], [3, 25, 32], [25, 33, 32], [32, 33, 31], [33, 31, 30], [33, 30, 29], [30, 29, 7], [3, 32, 4], [4, 5, 32], [31, 32, 5], [31, 30, 5], [30, 5, 6], [30, 6, 7]]

cartoon_expected_areas = [[0, 27, 31], [0, 1, 31], [1, 31, 32], [1, 12, 32], [12, 2, 32], [12, 23, 2], [2, 23, 33], [23, 29, 33], [29, 33, 34], [29, 30, 34], [30, 34, 28], [25, 22, 3], [25, 27, 3], [27, 31, 3], [31, 32, 3], [32, 2, 3], [2, 3, 4], [2, 4, 5], [2, 33, 5], [33, 34, 5], [5, 34, 28], [5, 28, 26], [5, 26, 24], [
    5, 9, 24], [4, 5, 9], [4, 8, 9], [4, 7, 8], [3, 7, 4], [3, 7, 22], [22, 20, 6], [6, 7, 22], [6, 7, 11], [7, 8, 11], [8, 11, 9], [9, 11, 10], [9, 10, 24], [10, 24, 21], [20, 6, 14], [6, 13, 14], [14, 13, 15], [13, 15, 16], [13, 16, 10], [16, 10, 21], [20, 14, 18], [14, 18, 17], [15, 18, 17], [15, 16, 17], [16, 17, 19], [16, 21, 19]]

human_expected_dis = [[11, 12], [12, 13], [14, 15], [15, 16], [21, 22], [23, 24], [17, 18], [17, 18], [17, 20], [18, 19], [18, 20], [18, 11], [18, 12], [18, 13],
                      [18, 21], [18, 22], [18, 14], [18, 15], [18, 16], [18, 23], [18, 24], [
                          18, 0], [18, 1], [18, 2], [18, 3], [18, 4], [18, 5], [18, 6],
                      [18, 7], [18, 8], [18, 9], [18, 10], [18, 25], [18, 26], [18, 27], [18, 28], [18, 29], [18, 34], [18, 32], [18, 31], [18, 30], [18, 33]]

cartoon_expected_dis = [[0, 1], [1, 12], [23, 29], [29, 30], [31, 32], [33, 34], [2, 4], [2, 3], [2, 5], [4, 3], [4, 5], [4, 0], [4, 1], [4, 12], [4, 31], [4, 32], [4, 32], [4, 29], [4, 30], [4, 33], [
    4, 34], [4, 27], [4, 25], [4, 22], [4, 20], [4, 18], [4, 17], [4, 19], [4, 21], [4, 24], [4, 26], [4, 28], [4, 6], [4, 7], [4, 8], [4, 9], [4, 10], [4, 11], [4, 14], [4, 15], [4, 16], [4, 13]]


# G:\Education\Semester 8\425-FYP\Dlib landmarks git\custom-dlib-shape-predictor-cartoon-human-mapping\Cartoon-front-face\
detector = dlib.cnn_face_detection_model_v1(
    r"face\mmod_human_face_detector.dat")
frontal_face_detector = dlib.get_frontal_face_detector()
cartoon_predictor = dlib.shape_predictor(model_path)
human_predictor = dlib.shape_predictor(human_model_path)
