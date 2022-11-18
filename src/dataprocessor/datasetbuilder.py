from src.imageprocessor.improc import *
import os


def build(url):
    print("start building dataset")
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(url)):
        if dirpath == url:
            continue

        print("building:", dirpath)

        for j, file in enumerate(filenames):
            try:
                image = cv2.imread(os.path.join(dirpath, file))
                faces = get_faces_inframe(image)

                for (x_start, y_start, dx, dy) in faces:
                    new_image = image[y_start:y_start+dy, x_start:x_start + dx]
                    cv2.imwrite(os.path.join(dirpath.replace('ny', 'new_val'), file), new_image)

                    break

            except Exception as exp:
                print(exp)


build('../../test/ny')
