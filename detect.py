import os
import argparse

import torch

PATH = os.path.dirname(os.path.realpath(__file__))

PATH_FINAL_TEST = os.path.join(PATH, 'final_test')

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
# load the yolo v5s classification algorithm with custom weights
model = torch.hub.load('ultralytics/yolov5', 'custom', path=os.path.join(PATH, 'best.pt')) 
# load the model on to the GPU
model.cuda()
model.to(device).eval()

def detect_and_classify_vrac(im_path):
    
    if im_path == 'all':
        for image in os.listdir(PATH_FINAL_TEST):
            if image.endswith(".jpg"):
                inference = model(os.path.join(PATH_FINAL_TEST, image))
                result = inference.pandas().xyxy[0]
                for i in result.index:
                    label = list(result['name'][i].values())[0]
                    box = [int(num) for num in list(result.iloc[i, 0:4])]
                    box[2] = box[2] - box[0]
                    box[3] = box[3] - box[1]
                    print('Image : ', image, '  |  Label : ', label, '  |  bbox : ', box)
                print("----------------------------")
                            
    else :
        inference = model(os.path.join(PATH_FINAL_TEST, im_path))
        result = inference.pandas().xyxy[0]
        for i in result.index:
            label = list(result['name'][i].values())[0]
            box = [int(num) for num in list(result.iloc[i, 0:4])]
            box[2] = box[2] - box[0]
            box[3] = box[3] - box[1]
            print('Image : ', im_path, '  |  Label : ', label, '  |  bbox : ', box)

def parse_args():
    
    parser = argparse.ArgumentParser(description='Food Detection')

    parser.add_argument('-p', '--path', dest='path', type=str, nargs=1,
                        help='Class name for the dataset, can be all to run on all images')
    
    args = parser.parse_args()

    if args.path is None:
        print("\nError in arguments, must give 1 image path !\n")
        exit()
        
    if args.path[0] == 'all':
        return args.path[0]
    
    if not os.path.exists(os.path.join(PATH_FINAL_TEST, args.path[0])):
        print("\nError in arguments, image doesn't exist in directory !\n, directory is : {PATH_FINAL_TEST}\n")
        exit()
        
    return args.path[0]

if __name__ == "__main__":
    
    args = parse_args()

    detect_and_classify_vrac(args)
