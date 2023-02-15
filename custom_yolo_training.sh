rm -rf yolov5

git clone https://github.com/ultralytics/yolov5
cd yolov5
pip3 install -r requirements.txt

wget https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s.pt -O yolov5s.pt

rm -rf runs/train/*

python3 train.py --img 640 --batch 8 --epochs 10 --data ../dataset_config.yaml --weights yolov5s.pt

cp runs/train/exp/weights/best.pt ../.