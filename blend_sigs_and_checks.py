import numpy as np
import os
from skimage import io
from skimage import color
from skimage import transform
from skimage import filters
from skimage import exposure
from skimage.util import crop
from random import shuffle
from preprocess.normalize import preprocess_signature
import matplotlib.pyplot as plt


def create_dir (save_path):

    if not os.path.exists(save_path):
        os.makedirs(save_path)


def create_file_structure(save_path, num_authors):

    create_dir(os.path.join(save_path, 'forgery'))
    create_dir(os.path.join(save_path, 'genuine'))
    for i in range(1, num_authors):
        create_dir(os.path.join(save_path, 'forgery', str(i).zfill(3)))
        create_dir(os.path.join(save_path, 'genuine', str(i).zfill(3)))


def load_signatures(load_path, sort=True):

    forgeries_authors_load_path = os.path.join(load_path, 'forgery')
    genuine_authors_load_path = os.path.join(load_path, 'genuine')

    forgeries_authors_list = os.listdir(forgeries_authors_load_path)
    genuine_authors_list = os.listdir(genuine_authors_load_path)

    if (sort):
        forgeries_authors_list.sort()
        genuine_authors_list.sort()

    return forgeries_authors_list, genuine_authors_list


def equals_size(reference, img, center):
    img = crop(
        img,
        ((center[0] - reference.shape[0]/2, img.shape[0] - center[0] - reference.shape[0]/2), 
        (center[1] - reference.shape[1]/2, img.shape[1] - center[1] - reference.shape[1]/2))
    )

    return img


def blend(foreground, background, method='multiply'):
    
    if foreground.shape != background.shape:
        raise Exception("Can't blend images with different shapes")

    if method != 'multiply':
        raise Exception('unknown blending method')

    blended = np.empty(background.shape, dtype='int32')

    for row in range(blended.shape[0]):
        for col in range(blended.shape[1]):
            if foreground[row][col] == 0:
                blended[row][col] = background[row][col]
            else:
                blended[row][col] = background[row][col] * foreground[row][col] / 255

    return blended


def blend_all(sigs, checks, checks_centers):
    
    checks_and_centers = list(zip(checks, checks_centers))
    shuffle(checks_and_centers)

    for index, sig in enumerate(sigs):
        sigs[index] = blend(
            sig,
            equals_size(
                sig,
                checks_and_centers[index % 20][0],
                checks_and_centers[index % 20][1]
            )
        )

    return sigs


def load_directory(path, format=None):
    
    if format != None and format.lower() not in ['png', 'jpeg', 'jpg']:
        raise Exception('invalid image format')

    imgs = list()
    files = os.listdir(path)
    files.sort()
    
    if format == None:
        for img_name in files:
            imgs.append(io.imread(os.path.join(path, img_name)))
    else:
        for img_name in files:
            if img_name[-len(format):].lower() == format.lower():
                imgs.append(io.imread(os.path.join(path, img_name)))

    return imgs


def rgb2gray_list(imgs_list):

    imgs_list = [color.rgb2gray(img) for img in imgs_list]

    return imgs_list


def preprocess_imgs_list(imgs_list, canvas_size):

    imgs_list = [preprocess_signature(img, canvas_size, input_size=(300, 480), img_size=(320, 502), invert=False) for img in imgs_list]

    return imgs_list


def rescale_intensity_imgs_list(imgs_list, in_range='image', out_range='dtype'):

    imgs_list = [exposure.rescale_intensity(img, in_range, out_range) for img in imgs_list]

    return imgs_list


if __name__ == '__main__':
    load_sigs = 'data/utsig' 
    load_checks = 'data/checks'
    save_sigs = 'data/background_utsig'

    canvas_size = (1627, 2387)
    c_col = [1800, 2602, 2067, 2713, 2077, 2638, 2287, 2195, 2071, 2164, 2890, 2264, 2300, 2055, 2785, 2586, 1286, 1285, 2472, 699]
    c_row = [1215, 659, 979, 1269, 834, 1087, 965, 767, 805, 836, 1189, 1140, 783, 1807, 1479, 452, 393, 376, 1877, 683]
    centers = list(zip(c_row, c_col))

    create_file_structure(save_sigs, 116)
    forgeries, genuine = load_signatures(load_sigs)

    checks = rescale_intensity_imgs_list(
        rgb2gray_list(load_directory(load_checks, 'jpg')),
        in_range=(0, 1),
        out_range=(0, 255)
    )
    for author in forgeries:
        author_sigs_path = os.path.join(
            load_sigs,
            'forgery',
            author
        )
        sigs = load_directory(author_sigs_path, 'png')
        sigs = preprocess_imgs_list(sigs, canvas_size)
        sigs = blend_all(sigs, checks, centers)
        
        for index, sig in enumerate(sigs):
            io.imsave(
                os.path.join(save_sigs, 'forgery', author, str(index) + '.png'),
                sig
            )

    for author in genuine:
        author_sigs_path = os.path.join(
            load_sigs,
            'genuine',
            author
        )
        sigs = load_directory(author_sigs_path, 'png')
        sigs = preprocess_imgs_list(sigs, canvas_size)
        sigs = blend_all(sigs, checks, centers)
        
        for index, sig in enumerate(sigs):
            io.imsave(
                os.path.join(save_sigs, 'genuine', author, str(index) + '.png'),
                sig
            )