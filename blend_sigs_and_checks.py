import numpy as np
import os
from skimage import io
from skimage import color
from skimage import transform
from skimage import filters
from skimage.util import crop
from random import shuffle


def create_dir (save_path):

    if not os.path.exists(save_path):
        os.makedirs(save_path)


def create_file_structure(save_path, num_authors):

    create_dir(os.path.join(save_path, 'forgery'))
    create_dir(os.path.join(save_path, 'genuine'))
    for i in range(1, num_authors):
        create_dir(os.path.join(save_path, 'forgery', str(i)))
        create_dir(os.path.join(save_path, 'genuine', str(i)))


def load_signatures(load_path, sort=True):

    forgeries_authors_load_path = os.path.join(load_path, 'forgery')
    genuine_authors_load_path = os.path.join(load_path, 'genuine')

    forgeries_authors_list = os.listdir(forgeries_authors_load_path)
    genuine_authors_list = os.listdir(genuine_authors_load_path)

    if (sort):
        forgeries_authors_list.sort()
        genuine_authors_list.sort()

    return forgeries_authors_list, genuine_authors_list


def equals_size(img1, img2, center):
    img2 = crop(
        img2,
        ((center[0] - img1.shape[0]/2, img2.shape[0] - center[0] - img1.shape[0]/2), 
        (center[1] - img1.shape[1]/2, img2.shape[1] - center[1] - img1.shape[1]/2))
    )

    return img2


def binarizes(img, max_val=1):

    if max_val != 1 and max_val != 255:
        raise Exception('Max value should be 1 or 255')

    threshold = filters.threshold_otsu(img, nbins=256)
    img[img <= threshold] = 0
    img[img > threshold] = max_val

    return img


def blend(foreground, background, method='multiply'):
    
    if foreground.shape != background.shape:
        raise Exception("Can't blend images with different shapes")

    if method != 'multiply':
        raise Exception('unknown blending method')

    binary_foreground = binarizes(foreground)

    blended = np.empty(background.shape, dtype='int32')

    for row in range(blended.shape[0]):
        for col in range(blended.shape[1]):
            if binary_foreground[row][col] == 0:
                blended[row][col] = background[row][col]
            else:
                blended[row][col] = background[row][col] * foreground[row][col] / 255


def blend_all(sigs, checks):
    shuffle(checks)

    for index, sig in enumerate(sigs):
        blend(sig, checks[index])



if __name__ == '__main__':
    load_sigs = 'data/utsig'
    load_checks = 'data/checks'
    save_sigs = 'data/background_utsig'

    create_file_structure(save_sigs, 116)
    forgeries, genuine = load_signatures(load_sigs)
    