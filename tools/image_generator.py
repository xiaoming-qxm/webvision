# -*- coding: utf-8 -*-
# Author: Xiaoming Qin


import os
import shutil
from os.path import join as pjoin
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img


def img_generator(datagen, src_image_path, dst_image_dir):
    img = load_img(src_image_path)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)

    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir=dst_image_dir,
                              save_format='jpg'):
        i += 1
        if i == 1:
            break


def traverse_folder_and_generate(source_folder, target_folder):
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)

    datagen = ImageDataGenerator(rotation_range=12,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 shear_range=0.1,
                                 horizontal_flip=True,
                                 vertical_flip=True,
                                 fill_mode='nearest')

    for class_folder in os.listdir(source_folder):
        src_class_path = source_folder + class_folder + '/'
        dest_class_path = target_folder + class_folder + '/'
        if not os.path.exists(dest_class_path):
            os.mkdir(dest_class_path)

        for img in os.listdir(src_class_path):
            src_img = src_class_path + img
            img_generator(datagen, src_img, dest_class_path)


def mk_img_fld():
    lst_root = "../data/over_samp"

    src_img_root = "/data/wv-40/train"
    dst_img_root = "/data/wo"

    for fname in os.listdir(lst_root):
        cls_id = fname.strip().split('.')[0]
        with open(pjoin(lst_root, fname), 'r') as f:
            lines = f.readlines()
        lines = [l.strip('\n') for l in lines]

        os.mkdir(pjoin(dst_img_root, cls_id))

        for i in range(len(lines)):
            shutil.copyfile(pjoin(src_img_root, cls_id, lines[i]),
                            pjoin(dst_img_root, cls_id,
                                  lines[i][:-4] + '_' + str(i) + '.jpg'))


if __name__ == "__main__":
    src_fld = "/data/wo/"
    dst_fld = "/data/wf/"
    traverse_folder_and_generate(src_fld, dst_fld)
    # mk_img_fld()
