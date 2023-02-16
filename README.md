# Custom Yolo Training & Inference for Food Detection

This tool is used to generate **custom datasets**, train a **Yolo** network on it and then run an **inference process**.
It is divided in 3 different steps that follow each other

- Step 1 : **Create adequate dataset**
- Step 2 : **Train the custom dataset**
- Step 3 : **Run an inference program**

# Remarks 

The training process can either be done on a computer with it's own GPU or on Google Colab

# Requirements

* Python >= **3.7**
* OpenCV
* PyTorch / Torch
* Yolov5 requirments ([link](https://github.com/ultralytics/yolov5/blob/master/requirements.txt))
* PIL, shutil, argparse, random, yaml, albumentions, split-folders
* ClearML install and init 

# How to run

1. Clone the repository and navigate to folder
  ```
  git clone https://gitlab.com/batoutou/trayvisor_test.git
  ```

2. Execute the 3 process in this following order :
  * To generate the dataset :

   ```
   python data_preprocessing.py -t [train option]
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

  * To run the tinference process :

   ```
   python detecty.py -p [img_name]
   ```

## Generate the dataset

In order to generate a custom dataset like the one for the legs it is recommended to use a software such as **Vott** ([installer link](https://github.com/Microsoft/VoTT/releases)) and annotate each image by hand.

If the desired dataset is similar to the handle one (small precise annotations of specific object) then you can use this custom dataset generator. To do so you will need several things first :

- templates : cropped image/photo in different orientation of the object
- backgrounds : random background that can be found anywhere, the more varied they are the better !

They should be added in the folder :

> /raw_data/"class_name"/templates
>
> /raw_data/"class_name"/backgrounds

With the "class_name" being the name of the object you wish to detect.

Then you have to run this command and replace :
- [name] by the "class_name" !
- [size] by the number of images you want in the end (keep in mind that after you'll have the augmentation that will add more images per class)
- [rotate] by **yes** or **no** wether or not you want your template images to be rotated from their original orientation

```
python3 preprocessed_dataset_generator.py -n [name] -s [size] -r [rotate]
```

    for example :

```
python3 preprocessed_dataset_generator.py -n handle -s 500 -r yes
```

This code will generate a new dataset with annotations composed of all combination between templates and backgrounds.
It will save the resulting images and labels in the folder :

> /custom_dataset/images/"class_name"
>
> /custom_dataset/labels/"class_name"

With the "class_name" being the name of the object you wish to detect.

## Augment a custom dataset

In order to augment a custom dataset like the one for the legs or/and handles, you can use this augmentation dataset generator. To do so you will need several things first :

- images : all images of all different objects you want to detect with size (640x480)
- labels : the corresponding labels for each image

They should be added in the folder

> /custom_dataset/images/"class_name"
>
> /custom_dataset/labels/"class_name"

With the "class_name" being the name of the object you wish to detect.

If you wish to detect several objects you must put images and labels for every objects.
Then you have to run this command and replace [names] by a list of "class_name" you want to detect and number by the number of augmented images you want per image of the class.

```
python3 dataset_augmentation_generator.py -n [names] -a [number]
```

    for example :

```
python3 dataset_augmentation_generator.py -n legs handle -a 40 10
```

This code will generate a new dataset with the original images and labels in addition to the new images and new labels that were generated.

It will save the resulting images and labels in the folder :

> /augmented_dataset

## Custom yolo training

1. First clone the 2 repositories that are used to train and make an engine :

```
git clone https://github.com/ultralytics/yolov5
```

```
git clone https://github.com/wang-xinyu/tensorrtx.git
```

Before running this script it is important to set your all settings :

- Update **CLASS_NUM** in tensorrtx/yolov5/yololayer.h if your model is trained on custom dataset
- Update **INPUT_H**  & **INPUT_W** in tensorrtx/yolov5/yololayer.h if your model is trained on specific image size
- Update a **line 5** of **custom_yolo_training.sh** : specify image size with only the biggest dimension of the image (--img 640), batch size (--batch 16), number of epochs (--epochs 15)
- In this example the script will use **Yolo V5s** version and the model will be trained with pretrained weights.
  ==> If you wish to change this please change the download link at **line 3** of of **custom_yolo_training.sh** with the desired weights.

Then you can run the script to train on the dataset you have created before :

```
sudo ./custom_yolo_training.sh
```

This will generate a engine file which is the Yolo engine.

> custom_yolov5s.engine

---

If you want to have a compatible engine with target (such as Jetson NX), please run this script first and then copy the following file to the target in addition to the following script :

> yolov5/yolov5s.wts
>
> custom_yolo_engine_target.sh

Then on target run the script and you should get also a custom engine file.

```
sudo ./custom_yolo_engine_target.sh
```
