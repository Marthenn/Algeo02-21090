import os.path
import sys

import yaml
from otf import *
from eigenface import eigenfaces


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
            format: (name, weight array, path image)
    """
    np.set_printoptions(threshold=sys.maxsize)

    face_dict = {}

    for i, el in enumerate(eigen_faces):
        key_name = 'eigen-{}'.format(i + 1)
        face_dict.update({key_name: str(el).replace('\n', '').replace('\\', '')})

    eigen_dict = {}

    eigen_dict.update({'mean-face': str(mean_face).replace('\n', '')})
    eigen_dict.update({'eigen-face': face_dict})

    if recognized_faces is None:
        return eigen_dict

    recogd_dict = {}

    for i, el in enumerate(recognized_faces):
        key_name = 'recognized-face-{}'.format(i + 1)
        dict_ins = {'name': el[0], 'weight': str(el[1]).replace('\n', '').replace('\\', ''), 'path': el[2]}
        recogd_dict.update({key_name: dict_ins})

    eigen_dict.update({'recognized-face': recogd_dict})

    return eigen_dict


# db yml generated from this file and has not been edited, not an arbitrary yml
# NOTE: please refer to doc/db-dictionary-doc.txt for the dictionary sample
def read_from_yml(path, file_name):
    with open(os.path.join(path, file_name)) as file:
        yml_dict = yaml.full_load(file)

    # parse mean-face arr
    mean_face = np.array(yml_dict.get('mean-face').replace('[', '').replace(']', '').replace('  ', ' ').split(' '))
    mean_face = mean_face[mean_face != ''].astype(float)

    yml_dict['mean-face'] = mean_face

    # parse eigenface
    raw_eigenface = yml_dict.get('eigen-face')

    eigen_list = None

    for i, el in enumerate(raw_eigenface):
        arr = np.array(
            yml_dict.get('eigen-face').get(el).replace('[', '').replace(']', '').replace('  ', ' ').split(' '))
        arr = arr[arr != ''].astype(float)

        if i == 0:
            eigen_list = np.empty((len(raw_eigenface), len(arr)))

        eigen_list[i] = arr

    yml_dict['eigen-face'] = eigen_list

    # parse recognized-face
    recogd_faces = yml_dict.get('recognized-face')

    recogd_list = []

    for i, el in enumerate(recogd_faces):
        face = recogd_faces.get(el)

        weight_arr = np.array(face.get('weight').replace('[', '').replace(']', '').replace('  ', ' ').split(','))
        weight_arr = weight_arr[weight_arr != ''].astype(float)

        tup = (face.get('name'), weight_arr, face.get('path'))

        recogd_list.append(tup)

    yml_dict['recognized-face'] = recogd_list

    return yml_dict


# sample code
if __name__ == '__main__':

    url = '../test/train'
    pict_arr = get_pict_array(url)
    folder_path = "/home/zidane/kuliah/Semester 3/IF2123 - Aljabar Linier dan Geometri/Algeo02-21090/eigenfaces/eigenface-{}.jpg"
    faces = eigenfaces(get_mean_diff_array(pict_arr, get_mean_vspace(pict_arr))).T
    mean_face = get_mean_vspace(pict_arr)
    recog_faces = [('zidane', [1, 2, 3, 4, 5, ], '127.0.0.1:8081/imek1.jpg'),
                    ('palkon', [1, 2, 4, 5, 67], '127.0.0.1:8081/imek1.jpg')]
    yml_dict = build_dict_eigen(mean_face, faces, recog_faces)
    save(yml_dict, "/home/zidane/kuliah/Semester 3/IF2123 - Aljabar Linier dan Geometri/Algeo02-21090")
    dict = read_from_yml("/home/zidane/kuliah/Semester 3/IF2123 - Aljabar Linier dan Geometri/Algeo02-21090", "db.yml")
    print(dict)

    # np.set_printoptions(threshold=sys.maxsize)
    # pict_arr = otf.get_pict_array('../test/train')
    # mean_face = otf.get_mean_vspace(pict_arr)
    # eigen_faces = np.arange(100).reshape(10, 10)
    #
    # arr = np.array([1, 2, 10, 2, 12, 2])
    # print(arr[arr != 2])
    #
    # recog_faces = [('zidane', [1, 2, 3, 4, 5, ], '127.0.0.1:8081/imek1.jpg'),
    #                ('palkon', [1, 2, 4, 5, 67], '127.0.0.1:8081/imek1.jpg')]
    # dict = build_dict_eigen(mean_face, eigen_faces, recog_faces)
    #
    # save(dict, '../test/train')
    # read_from_yml('../test/train', 'db.yml')
