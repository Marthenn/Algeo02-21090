import numpy as np

__image_dim = 80


def normalize_image_value(arr):
    return (255 * (arr - np.min(arr)) / np.ptp(arr)).astype(float)


def get_image_dim():
    return __image_dim

# sample code for saving pict, normalized

# url ='../test/train'
# pict_arr = get_pict_array(url)
# folder_path = "/home/zidane/kuliah/Semester 3/IF2123 - Aljabar Linier dan Geometri/Algeo02-21090/eigenfaces/eigenface-{}.jpg"
# faces = eigenfaces(get_mean_diff_array(pict_arr, get_mean_vspace(pict_arr))).T
# mean_face = get_mean_vspace(pict_arr)
#
# for i, el in enumerate(faces):
#     filename = 'eigenface-{}'.format(i)
#     el = util.normalize_image_value(el)
#
#     face = el.reshape(80, 80)
#
#     cv2.imwrite(folder_path.format(i).format(i),
#                 face)
