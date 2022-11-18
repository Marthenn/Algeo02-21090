import cv2
import math
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

face_cascade = cv2.CascadeClassifier('/home/zidane/kuliah/Semester 3/IF2123 - Aljabar Linier dan Geometri/Algeo02-21090/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/home/zidane/kuliah/Semester 3/IF2123 - Aljabar Linier dan Geometri/Algeo02-21090/haarcascades/haarcascade_eye.xml')


# TODO: GANTI
def euclidean_distance(a, b):
    x1 = a[0];
    y1 = a[1]
    x2 = b[0];
    y2 = b[1]
    return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))


def preprocess_image(image):
    image = align_face(image)
    return image


def align_face(image_frame):
    face_cascade_name = cv2.data.haarcascades + 'haarcascade_eye.xml'
    face_cascadex = cv2.CascadeClassifier()
    face_cascadex.load(cv2.samples.findFile(face_cascade_name))
    eyes = eye_cascade.detectMultiScale(image_frame, 1.5, 4)

    eye_atas = (0, 0)
    eye_bawah = (0, 0)

    for i, (x_start, y_start, dx, dy) in enumerate(eyes):
        if eye_atas == (0, 0):
            eye_atas = (int(x_start + dx / 2), int(y_start + dy / 2))
        else:
            eye_bawah = (int(x_start + dx / 2), int(y_start + dy / 2))

        if i == 1:
            break

    if eye_atas[1] < eye_bawah[1]:
        eye_atas, eye_bawah = eye_bawah, eye_atas

    # decide which point is higher
    point_atas = eye_atas
    point_bawah = eye_bawah
    clockwise = 0

    if point_atas[1] < point_bawah[1]:
        point_atas, point_bawah = point_bawah, point_atas

    if eye_atas[0] > eye_bawah[0]:
        clockwise = 1
    else:
        clockwise = -1

    point_o = (point_atas[0], point_bawah[1])

    angle = math.atan(euclidean_distance(point_atas, point_o) / euclidean_distance(point_bawah, point_o))
    angle *= 180
    angle /= math.pi

    aligned_face = np.array(Image.fromarray(image_frame).rotate(angle * clockwise))

    faces = face_cascade.detectMultiScale(aligned_face, 1.08, 5, minSize=(150, 150))

    for (x_start, y_start, dx, dy) in faces:
        aligned_face = aligned_face[y_start - 20:dy + y_start + 20, x_start - 20:x_start + dx + 20]
        break

    return aligned_face


def get_faces_inframe(rgb_frame):
    grey_scale_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
    # test for min size
    faces = face_cascade.detectMultiScale(grey_scale_frame, 1.03, 10, minSize=(100, 100))

    return faces


def hist_eq(frame):
    h = frame.shape[0]
    w = frame.shape[1]

    face = cv2.equalizeHist(frame)
    mid_point = math.floor(w/2)

    left_face = cv2.equalizeHist(frame[:, :mid_point])
    right_face = cv2.equalizeHist(frame[:, mid_point:])

    for y in range(h):
        for x in range(w):
            if x < w / 4:
                v = left_face[y][x]
            elif x < math.floor(w * 2 / 4):
                lv = left_face[y][x]
                wv = face[y][x]
                f = (x - w * 1 / 4) / float(w / 4)
                v = round((1.0 - f) * lv + f * wv)
            elif x < w * 3 / 4:
                rv = right_face[y][x-mid_point]
                wv = face[y][x]
                f = (x - w * 2 / 4) / float(w / 4)
                v = round((1.0 - f) * wv + f * rv)
            else:
                v = right_face[y][x - mid_point]

            frame[y][x] = v

    # cv2.imwrite(
    #     '/home/zidane/kuliah/Semester 3/IF2123 - Aljabar Linier dan Geometri/Algeo02-21090/test/new_train/Adriana_Lima/adriana_limax.jpg',
    #     frame)

    return frame


if __name__ == '__main__':
    image = cv2.imread(
        '/home/zidane/kuliah/Semester 3/IF2123 - Aljabar Linier dan Geometri/Algeo02-21090/test/new_train/Adriana_Lima/adriana_lima0.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cv2.imwrite(
        '/home/zidane/kuliah/Semester 3/IF2123 - Aljabar Linier dan Geometri/Algeo02-21090/test/new_train/Adriana_Lima/adriana_limay.jpg',
        image)
    hist_eq(image)
