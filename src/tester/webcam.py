from src.imageprocessor.improc import *
from src import recog
from src.dataprocessor import ymldb


def start_webcam():
    data = ymldb.read_from_yml("/", "db.yml")
    tresh = recog.get_treshold(data)

    retry = False
    retry_count = 0
    frame_count = 0
    camera = cv2.VideoCapture(0)
    name = 'unknown'

    # keep camera alive
    while cv2.waitKey(1) == -1:
        available, rgb_frame = camera.read()

        if not available:
            continue

        frame_count += 1

        # test for min size
        faces = get_faces_inframe(rgb_frame)
        for (x_start, y_start, dx, dy) in faces:
            if frame_count % 50 == 0 or (retry is True and retry_count <= 50) or cv2.waitKey(1) == ord('s'):
                # add pixels so we won't get images with black corners
                face_frame = rgb_frame[y_start - 80:, x_start - 80:x_start + dx + 80]
                try:
                    # cv2.imwrite("/home/zidane/kuliah/Semester 3/IF2123 - Aljabar Linier dan Geometri/b4eye.jpg",
                    #             rgb_frame[y_start:, x_start:x_start + dx])
                    print('pre-processing face')
                    processed_image = align_face(face_frame)

                    if processed_image is None:
                        retry = True
                        retry_count += 1
                        print('retry none')
                        continue

                    print('face aligned!')
                    retry = False
                    retry_count = 0

                    res = recog.find_match(cv2.cvtColor(processed_image, cv2.COLOR_RGB2GRAY),
                                           data, tresh)
                    if res is not None:
                        name = res[0]
                    else:
                        if res is not None:
                            res = recog.find_match(cv2.cvtColor(rgb_frame[y_start:, x_start:x_start + dx], cv2.COLOR_RGB2GRAY),data, tresh)
                            name = res[0]
                        else:
                            name = 'unknown'

                except cv2.error as exp:
                    retry = True
                    retry_count += 1
                    retry_count %= 100
                    print(exp)

                except ZeroDivisionError as exp:
                    retry = True
                    retry_count += 1
                    retry_count %= 100
                    print(exp)

            cv2.rectangle(rgb_frame, (x_start, y_start), (x_start + dx, y_start + dy), (255, 0, 0), 2)
            cv2.putText(rgb_frame, name, (x_start, y_start - 12), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow('pres ss to quit', rgb_frame)


if __name__ == '__main__':
    start_webcam()
