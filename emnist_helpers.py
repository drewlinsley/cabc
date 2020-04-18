import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
import scipy
from scipy import ndimage
import skimage
from skimage.morphology import medial_axis
import time
import cv2
import random
import os
import sys
from scipy import linalg as LA

def find_files(files, dirs=[], contains=[]):
    for d in dirs:
        onlyfiles = [os.path.join(d, f) for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]
        for i, part in enumerate(contains):
            files += [os.path.join(d, f) for f in onlyfiles if part in f]
        onlydirs = [os.path.join(d, dd) for dd in os.listdir(d) if os.path.isdir(os.path.join(d, dd))]
        if onlydirs:
            recursive_files, _ = find_files([], onlydirs, contains)
            files += recursive_files
    return files, len(files)


def make_list(load_root='/Users/junkyungkim/Desktop/by_class/', save_name='list.npy',
              only_include=[]):
    class_names = [class_name for class_name in os.listdir(load_root) if
                   os.path.isdir(os.path.join(load_root, class_name))]
    class_path_list = [os.path.join(load_root, class_name) for class_name in os.listdir(load_root) if
                       os.path.isdir(os.path.join(load_root, class_name))]
    if len(only_include) > 0:
        for i, cl in enumerate(class_names):
            if cl not in only_include:
                class_path_list.remove(os.path.join(load_root, cl))
    ims_list = []
    num_ims_list = []
    for i, iclass in enumerate(class_path_list):
        print('class ' + str(i))
        out_tuple = find_files([], dirs=[iclass], contains=['.png'])
        ims_list.append(out_tuple[0])
        num_ims_list.append(out_tuple[1])
    np.save(os.path.join(load_root,save_name),(ims_list, num_ims_list))
    # (2,62) object. (paths, num_ims) x categories


def load_list(load_fn):
    npy = np.load(load_fn)
    # (2,62) object. (paths, num_ims) x categories
    return npy[0], npy[1]

if __name__ == "__main__":
    # only capitals
    capitals = ['4a', '4b', '4c', '4e', '4f',
                '5a', '6c', '6f',
                '41', '42', '43', '44', '45', '47', '48', '49',
                '50', '51', '52', '53', '54', '55', '56', '57', '58']
    numbers = ['30', '31', '32', '33', '34', '35', '36', '37', '38', '39']
    nist_root = str(sys.argv[1])
    make_list(nist_root, only_include=capitals)