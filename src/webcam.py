import cv2
import math
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

# TODO: Multi-threaded optimization to achieve better fps

face_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_eye.xml')


def euclidean_distance(a, b):
    x1 = a[0];
    y1 = a[1]
    x2 = b[0];
    y2 = b[1]
    return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))


# CAUTION: image not mirror'd
def align_face(image_frame):
    eyes = eye_cascade.detectMultiScale(image_frame, 1.05, 5)

    if len(eyes) != 2:
        return None

    eye_atas = (0, 0)
    eye_bawah = (0, 0)

    for (x_start, y_start, dx, dy) in eyes:
        if eye_atas == (0, 0):
            eye_atas = (int(x_start + dx / 2), int(y_start + dy / 2))
        else:
            eye_bawah = (int(x_start + dx / 2), int(y_start + dy / 2))

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
        aligned_face = aligned_face[y_start-20:dy + y_start+20, x_start-20:x_start + dx+20]
        break

    return aligned_face


def start_webcam():
    retry = False
    retry_count = 0
    frame_count = 0
    camera = cv2.VideoCapture(0)

    # keep camera alive
    while cv2.waitKey(1) == -1:
        available, rgb_frame = camera.read()

        if not available:
            continue

        frame_count += 1

        grey_scale_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
        # test for min size
        faces = face_cascade.detectMultiScale(grey_scale_frame, 1.05, 5, minSize=(100, 100))
        for (x_start, y_start, dx, dy) in faces:
            if frame_count % 100 == 0 or (retry is True and retry_count <= 50):
                # add pixels so we won't get images with black corners
                face_frame = rgb_frame[y_start - 80:, x_start - 80:x_start + dx + 80]
                try:
                    cv2.imwrite("/home/zidane/kuliah/Semester 3/IF2123 - Aljabar Linier dan Geometri/b4eye.jpg",
                                face_frame)
                    print('pre-processing face')
                    processed_image = preprocess_image(face_frame)

                    if processed_image is None:
                        retry = True
                        retry_count += 1
                        print('retry none')
                        continue

                    cv2.imwrite("/home/zidane/kuliah/Semester 3/IF2123 - Aljabar Linier dan Geometri/camframe1.jpg",
                                processed_image)

                    print('face aligned!')
                    retry = False
                    retry_count = 0

                except cv2.error as exp:
                    print(processed_image)
                    retry = True
                    retry_count += 1
                    retry_count %= 100
                    print(exp)

            cv2.rectangle(rgb_frame, (x_start, y_start), (x_start + dx, y_start + dy), (255, 0, 0), 2)
            cv2.putText(rgb_frame, 'Zidane', (x_start, y_start - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow('webcam', rgb_frame)


def preprocess_image(image):
    image = align_face(image)
    blurred_image = gaussian_filter(image, sigma=3)
    return blurred_image


if __name__ == '__main__':
    start_webcam()