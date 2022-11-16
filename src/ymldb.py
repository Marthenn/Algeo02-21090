import os.path
import sys
from collections import OrderedDict

import numpy as np
import yaml
import otf


def save(dictionary, output_dir):
    filename = 'db.yml'

    with open(os.path.join(output_dir, filename), 'w') as file:
        yaml.dump(dictionary, file, sort_keys=False)


def build_dict_eigen(mean_face, eigen_faces, recognized_faces=None):
    """
        mean_face format: 1d numpy array

        eigen_faces = 2d numpy array, each row represents an eigen face

        recognized faces format:
            type: tuple
            format: (name, weight array)

    """

    face_dict = {}

    for i, el in enumerate(eigen_faces):
        key_name = 'eigen-{}'.format(i + 1)
        face_dict.update({key_name: str(el)})

    eigen_dict = {}

    eigen_dict.update({'mean-face': str(mean_face).replace('\n', '')})
    eigen_dict.update({'eigen-face': face_dict})

    if recognized_faces is None:
        return eigen_dict

    recogd_dict = {}

    for i, el in enumerate(recognized_faces):
        key_name = 'recognized-face-{}'.format(i + 1)
        dict_ins = {'name': el[0], 'weight': str(el[1])}
        recogd_dict.update({key_name: dict_ins})

    eigen_dict.update({'recognized-face': recogd_dict})

    return eigen_dict


# sample code
if __name__ == '__main__':
    np.set_printoptions(threshold=sys.maxsize)
    pict_arr = otf.get_pict_array('../test/train')
    mean_face = otf.get_mean_vspace(pict_arr)
    eigen_faces = np.arange(100).reshape(10, 10)

    recog_faces = [('zidane', [1, 2, 3, 4, 5, ]), ('palkon', [1, 2, 4, 5, 67])]
    dict = build_dict_eigen(mean_face, eigen_faces, recog_faces)

    save(dict, '../test/train')
