# Custom Yolo Training & Inference for Food Detection

This tool is used to generate **custom datasets**, train a **Yolo** network on it and then run an **inference process**

It is divided in 3 different steps that follow each other

- Step 1 : **Create an adequate dataset**
- Step 2 : **Train the custom dataset**
- Step 3 : **Run an inference program**

# Remarks 

The training process can either be done on a computer with it's own GPU or on Google Colab

# Requirements

* Python >= **3.7**
* OpenCV
* PyTorch / Torch
* Yolov5 requirements ([link](https://github.com/ultralytics/yolov5/blob/master/requirements.txt))
* shutil, argparse, yaml, split-folders

# How to run

1. Clone the repository and navigate to folder
  ```
  git clone https://gitlab.com/batoutou/trayvisor_test.git
  ```

2. Execute the 3 processes in this following order :
  * To generate the dataset :

   ```
   python data_preprocessing.py -t [train]
   ```

  * To train on a local GPU or Google Colab :
  
   For the local GPU training
   ```
   bash custom_yolo_training.sh
   ```

   For the Google Colab training
   ```
   Import 'dataset.zip' into the 'YOLOv5-Custom-Training.ipynb'
   ```

   ```
   Execute the notebook
   ```

  * To run the inference process :

   ```
   python detecty.py -p [img_name]
   ```

## Generate the dataset

In order to generate the dataset so that the Yolo V5 training script can understand it, I had to create a first python file that would do 5 things :

- Read the JSON file : get all the information from the JSON file and store it as a dictionary
- Create a file for each image : for each image, it creates a '.txt' file with the Yolo V5 format (bounding box are represented as a ratio between 0 and 1)
  > [class] [x_center] [y_center] [width] [height]
- Train, Val, Test split : the images and their labels are separated into 3 folders for the training, validation and testing. In addition, I left 10 images on the side for the inference process
- The config.yaml file is generated either for local gpu or google colab training
- Zip file for Google Colab : A zip file containing the dataset and the Yaml file is created for training on Google Colab

To execute, you have to run this command and replace :
* [train] by either **gpu** or **colab** depending on what training you want
```
python data_preprocessing.py -t [train]
```

for example :
```
python data_preprocessing.py -t gpu
```

## Training the custom model

In order to train the custom Yolo V5 model we have two options as mentioned before. 

* Train on local GPU using a shell script I made
 ```
 bash custom_yolo_training.sh
 ```
 In this file at the **line 11** you can fine tune the hyper parameters of the training : 
 - max image size
 - number of epochs
 - batch size

 This script will output the best weights of the training, which you can latter use for inference
 > best.pt

* Train on Google Colab
 - First, you'll need to import the **dataset.zip** file into the colab notebook **YOLOv5-Custom-Training.ipynb** 
 - Than you can execute the notebook which in the end will download the best weights that you will use for inference


## Run an inference program

Run the following command to launch the program to detect wether it is a small vrac or a big vrac

* The images you want to detect need to be in the **final_test** folder 
* replace [im_path] with the image name, no need to put the full path
* You can replace [im_path] by **all** if you want to run inference on all of them

```
python detect.py -p [im_path]
```

for example :
```
python detect.py -p im_0.jpg
```

This will output the predicted labels and bounding box for the image
