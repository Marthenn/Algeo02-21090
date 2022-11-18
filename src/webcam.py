from src.imageprocessor.improc import *


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

        # test for min size
        faces = get_faces_inframe(rgb_frame)
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
                    retry = True
                    retry_count += 1
                    retry_count %= 100
                    print(exp)

            cv2.rectangle(rgb_frame, (x_start, y_start), (x_start + dx, y_start + dy), (255, 0, 0), 2)
            cv2.putText(rgb_frame, 'Zidane', (x_start, y_start - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow('webcam', rgb_frame)


if __name__ == '__main__':
    start_webcam()
