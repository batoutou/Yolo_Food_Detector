import os
import argparse

import torch

PATH = os.path.dirname(os.path.realpath(__file__))

PATH_FINAL_TEST = os.path.join(PATH, 'final_test')

model = torch.hub.load('ultralytics/yolov5', 'custom', path=os.path.join(PATH, 'best.pt')) 

def detect_and_classify_vrac(im_path):
    
    inference = model(im_path)
    
    print(inference.pandas().xyxy[0])

def parse_args():
    
    parser = argparse.ArgumentParser(description='Food Detection')

    parser.add_argument('-p', '--path', dest='path', type=str, nargs=1,
                        help='Class name for the dataset')
    
    args = parser.parse_args()

    if args.path is None:
        print("\nError in arguments, must give 1 image path !\n")
        exit()
    
    if not os.path.exists(os.path.join(PATH_FINAL_TEST, args.path[0])):
        print("\nError in arguments, image doesn't exist in directory !\n, directory is : {PATH_FINAL_TEST}\n")
        exit()

    return os.path.join(PATH_FINAL_TEST, args.path[0])

if __name__ == "__main__":
    
    args = parse_args()

    detect_and_classify_vrac(args)
