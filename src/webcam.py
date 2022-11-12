import cv2
import math
import numpy as np
from PIL import Image

# TODO: Multi-threaded optimization to achieve better fps

face_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_eye.xml')

camera = cv2.VideoCapture(0)
frame_count = 0


def euclidean_distance(a, b):
    x1 = a[0];
    y1 = a[1]
    x2 = b[0];
    y2 = b[1]
    return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))


# CAUTION: image not mirror'd
def align_face(image_frame_gs):
    eyes = eye_cascade.detectMultiScale(image_frame_gs, 1.3, 6)

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

    aligned_face = np.array(Image.fromarray(image_frame_gs).rotate(angle * clockwise))

    faces = face_cascade.detectMultiScale(aligned_face, 1.05, 5, minSize=(200, 200))

    for (x_start, y_start, dx, dy) in faces:
        aligned_face = aligned_face[y_start:dy + y_start, x_start:x_start + dx]
        break

    return aligned_face


import threading

retry = False
retry_count = 0

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
        if frame_count % 100 == 0 or (retry is True and retry_count <= 30):
            face_frame = rgb_frame[y_start - 60:dy + y_start + 60, x_start - 60:x_start + dx + 60]
            try:
                cv2.imwrite("/home/zidane/kuliah/Semester 3/IF2123 - Aljabar Linier dan Geometri/b4eye.jpg", face_frame)
                print('aligning face')
                aligned_face = align_face(face_frame)

                if aligned_face is None:
                    retry = True
                    retry_count += 1
                    retry_count %= 100
                    print('retry none')
                    continue

                cv2.imwrite("/home/zidane/kuliah/Semester 3/IF2123 - Aljabar Linier dan Geometri/camframe.jpg",
                            aligned_face)

                print('face aligned!')
                retry = False
                retry_count = 0

            except cv2.error:
                retry = True
                retry_count += 1
                retry_count %= 100
                print('retry exception')

        cv2.rectangle(rgb_frame, (x_start, y_start), (x_start + dx, y_start + dy), (255, 0, 0), 2)
        cv2.putText(rgb_frame, 'Zidane', (x_start, y_start - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow('webcam', rgb_frame)
