# untuk testing accuraccy dan pilih threshold
import os
from recog import *

def test_accuracy(multiplier,data, treshold):
    print("Testing accuracy with multiplier = {}".format(multiplier))
    folder_path = 'D:/Kuliah/Semester 3/AlGeo/Algeo02-21090/test/val'
    true = 0
    total = 0
    tres = treshold * multiplier
    print("Treshold = {}".format(tres))
    # print("folder path = {}".format(folder_path))
    for i, (dirpath, dirnames, filename) in enumerate(os.walk(folder_path)):
        if dirpath == folder_path:
            continue
        # print('Processing {}'.format(os.path.basename(os.path.normpath(dirpath))))
        for j, pic in enumerate(filename):
            # print('Processing {}'.format(pic))
            if pic.endswith(".jpg") or pic.endswith(".png") or pic.endswith(".jpeg"):
                res = find_match(os.path.join(dirpath, pic), data, tres)
                if res != None: 
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
    print("Accuracy: {}".format(true/total))

if __name__ == '__main__':
    data = read_from_yml("D:\Kuliah\Semester 3\AlGeo\Algeo02-21090", "db.yml")
    treshold = get_treshold(data, 1)
    for i in range(5,105,5):
        i/=100
        accuracy = test_accuracy(i,data,treshold)
