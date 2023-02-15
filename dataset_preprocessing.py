import cv2
import os
import sys

import json
from zipfile import ZipFile
import shutil
import splitfolders
import yaml

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# training on GPU or on Google Colab
GPU = False 
COLAB = True

PATH = os.path.dirname(os.path.realpath(__file__))

# set path for images, labels and final dataset
PATH_IMG = os.path.join(PATH, 'images')
PATH_LBL = os.path.join(PATH, 'labels')
PATH_JSON = os.path.join(PATH, 'trayvisor_test_db.json')

PATH_DATASET = os.path.join(PATH, 'dataset')
PATH_COLAB = '/content/dataset'

PATH_FINAL_TEST = os.path.join(PATH, 'final_test')

final_test_size = 10
classes = ['small_vrac', 'big_vrac']

try:
    shutil.rmtree(PATH_IMG)
    shutil.rmtree(PATH_LBL)
    shutil.rmtree(PATH_DATASET)
    shutil.rmtree(PATH_FINAL_TEST)
except:
    pass

os.makedirs(PATH_LBL)
os.makedirs(PATH_DATASET)
os.makedirs(PATH_FINAL_TEST)
os.makedirs(os.path.join(PATH, 'tmp_img'))
os.makedirs(os.path.join(PATH, 'tmp_lbl'))

with ZipFile(os.path.join(PATH, 'images.zip'), 'r') as f:
    f.extractall()
    
# load json
with open(PATH_JSON, 'r') as file:
    json_dict = json.load(file)
file.close()
 
cmt = 0
# iterate through folder to get images and classes
for image in os.listdir(PATH_IMG):
    
    img_path = os.path.join(PATH_IMG, image)
    img_lbl = image.split('.')[0] + '.txt'
    img_lbl_path = os.path.join(PATH_LBL, img_lbl)
        
    img = cv2.imread(img_path)
    
    height, width = img.shape[:2]

    boxes = []
    labels = []
    
    img_lbl = image.split('.')[0] + '.txt'
    img_lbl_path = os.path.join(PATH_LBL, img_lbl)
    
    if os.path.exists(img_lbl_path):
        continue
    
    for image_dict in json_dict:
        if image_dict['name'] == image:
            for box in image_dict['boxes']:
                color = ()
                if box['id'] == 'big_vrac':
                    labels.append(1)
                    color = (255, 0, 0)
                else:
                    labels.append(0)
                    color = (0, 255, 0)

                x = box['box'][0]
                y = box['box'][1]
                w = box['box'][2]
                h = box['box'][3]

                boxes.append([(x+(w/2))/width, (y+(h/2))/height, w/width, h/height])
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 4)
                cv2.putText(img, box['id'], (x+5, y+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 1)

    # To visualize the image with each bounding box
    # cv2.imshow("image", img)
    # cv2.waitKey(0)
            
        with open(img_lbl_path, 'w') as file:
            for i in range(len(boxes)):
                file.write(str(labels[i]) + ' ')

                for j in range(len(boxes[i])):
                    file.write(str(boxes[i][j]) + ' ')

                file.write('\n')

        file.close()
    
    if cmt < final_test_size:
        shutil.move(img_path, PATH_FINAL_TEST)
        shutil.move(img_lbl_path, PATH_FINAL_TEST)
        cmt += 1
        continue

shutil.copytree(PATH_IMG, os.path.join(PATH, 'tmp_img', 'images'), copy_function = shutil.copy)
shutil.copytree(PATH_LBL, os.path.join(PATH, 'tmp_lbl', 'labels'), copy_function = shutil.copy)

splitfolders.ratio(os.path.join(PATH, 'tmp_img'), PATH_DATASET, seed=1337, ratio=(.75, 0.15,0.1), group_prefix=None, move=True) 
splitfolders.ratio(os.path.join(PATH, 'tmp_lbl'), PATH_DATASET, seed=1337, ratio=(.75, 0.15,0.1), group_prefix=None, move=True) 

shutil.rmtree(os.path.join(PATH, 'tmp_img'))
shutil.rmtree(os.path.join(PATH, 'tmp_lbl'))

# create the Yaml configuration file por the Yolo training    
names = []

for i, class_name in enumerate(classes):
    dict_name = {i: class_name}
    names.append(dict_name)

if GPU is True:
    dataset_config = {
            'path': PATH_DATASET,
            'train': 'train',
            'val': 'val',
            'test': 'test',
            'names': names,
            'download': None
    }

elif COLAB is True:
    dataset_config = {
            'path': PATH_COLAB,
            'train': 'train',
            'val': 'val',
            'test': 'test',
            'names': names,
            'download': None
    }

with open(os.path.join(PATH_DATASET, "dataset_config.yaml"), 'w') as yamlfile:
    data = yaml.dump(dataset_config, yamlfile, sort_keys=False)
    
shutil.make_archive('dataset', 'zip', PATH_DATASET)
