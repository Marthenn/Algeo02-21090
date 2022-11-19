# untuk testing accuraccy dan pilih threshold
import os
from recog import *


def test_accuracy(data, treshold):
    print("Testing accuracy with treshold = {}".format(treshold))
    folder_path = '../test/archive(1)'
    true = 0
    total = 0
    tres = treshold
    print("Treshold = {}".format(tres))
    # print("folder path = {}".format(folder_path))
    for i, (dirpath, dirnames, filename) in enumerate(os.walk(folder_path)):

        # print('Processing {}'.format(os.path.basename(os.path.normpath(dirpath))))
        for j, pic in enumerate(filename):
            # print('Processing {}'.format(pic))
            if pic.endswith(".jpg") or pic.endswith(".png") or pic.endswith(".jpeg") or pic.endswith(".pgm"):
                res = find_match(os.path.join(dirpath, pic), data, tres)
                if res != None:
                    print(os.path.basename(os.path.normpath(dirpath)), res[0], 'path')
                    if res[0] == os.path.basename(os.path.normpath(dirpath)):
                        true += 1
                total += 1
    # for (dirpath, dirnames, filename) in (os.walk(folder_path)):
    #     if dirpath == folder_path:
    #         continue
    #     print('Processing {}'.format(os.path.basename(os.path.normpath(dirpath))))
    #     for j, pic in enumerate(filename):
    #         print('Processing {}'.format(pic))
    #         if pic.endswith(".jpg") or pic.endswith(".png") or pic.endswith(".jpeg"):
    #             res = find_match(os.path.join(dirpath, pic), data, tres)
    #             if res!=None:
    #                 if res[0] == os.path.basename(os.path.normpath(dirpath)):
    #                     true += 1
    #             total+=1
    print("true: {} total: {} Accuracy: {}".format(true, total, true / total))


if __name__ == '__main__':
    data = read_from_yml("/home/zidane/kuliah/Semester 3/IF2123 - Aljabar Linier dan Geometri/Algeo02-21090", "db.yml")
    tresh = get_treshold(data)
    if tresh == 0:
        tresh = 500
    test_accuracy(data, tresh)
