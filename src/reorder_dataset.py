'''
Re-ordering dataset directory as required by facenet

Current working directory : 
    dataset
    |-- 1.png
    |-- 2.png
    |-- 3.png
    '-- n.png

Required working directory : 
    dataset
    |-- airplane
    |   |-- airplane_0001.png
    |   |-- airplane_0002.png
    |   '-- airplane_0003.png
    |-- cat
    |   |-- cat_0001.png
    |   |-- cat_0002.png
    |   '-- cat_0003.png
    |-- frog
    |   |-- frog_0001.png
    |   |-- frog_0002.png
    |   '-- frog_0003.png
    '-- truck
        |-- truck_0001.png
        |-- truck_0002.png
        '-- truck_0003.png
'''

import pandas as pd
import os
import cv2
from tqdm import tqdm
import argparse

def main():
    parser = argparse.ArgumentParser(description="Parser to create tfrecords")
    parser.add_argument("--csv", type=str, required=True,
                                    help="cifar10 annotation filename with .csv extensions")
    parser.add_argument("--dataset", type=str, required=True,
                                    help="path to image dataset")
    parser.add_argument("--new-dir", type=str, default='stacked_train',
                                    help="path to the new dataset directory")
    parser.add_argument("--extensions", type=str, default='.png',
                                    help="image extensions")

    args = parser.parse_args()

    csv_annotation = args.csv
    data_dir = args.dataset
    stacked_dir = args.new_dir
    image_ext = args.extensions

    data = pd.read_csv(csv_annotation)
    idx = list(data['id'])
    classes = list(data['label'])

    if not os.path.exists(stacked_dir):
        os.mkdir(stacked_dir)

    for i in tqdm(idx, 'Processing'):
        stacked_path = os.path.join(stacked_dir, classes[i-1])
        if not os.path.exists(stacked_path):
            os.mkdir(stacked_path)
        img = cv2.imread(os.path.join(data_dir, str(i)) + image_ext)
        filename = classes[i-1] + '_' + str(i).zfill(4) + image_ext
        cv2.imwrite(os.path.join(stacked_path, filename), img)

if __name__ == "__main__":
    main()