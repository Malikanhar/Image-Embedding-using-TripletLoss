import numpy as np
import pandas as pd
import argparse
import pickle
import os
import imageio
from tqdm import tqdm

def unpickle(data_dir, file):
    with open(os.path.join(data_dir, file), 'rb') as fo:
        res = pickle.load(fo, encoding='bytes')
    return res

def main():
    parser = argparse.ArgumentParser(description='Parser to extract Cifar100 dataset')
    parser.add_argument('--data_dir', type=str, required=True,
                                    help='Data directory consisting of meta, train and test file')
    args = parser.parse_args()
    data_dir = args.data_dir
    meta = unpickle(data_dir, 'meta')

    fine_label_names = [t.decode('utf8') for t in meta[b'fine_label_names']]

    train = unpickle(data_dir, 'train')

    filenames = [t.decode('utf8') for t in train[b'filenames']]
    fine_labels = train[b'fine_labels']
    data = train[b'data']

    images = list()
    for d in data:
        image = np.zeros((32,32,3), dtype=np.uint8)
        image[...,0] = np.reshape(d[:1024], (32,32)) # Red channel
        image[...,1] = np.reshape(d[1024:2048], (32,32)) # Green channel
        image[...,2] = np.reshape(d[2048:], (32,32)) # Blue channel
        images.append(image)

    img_count = 1
    ids = []
    labels = []
    img_dir = os.path.join(data_dir, 'image')
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    for index,image in tqdm(enumerate(images)):
        filename = os.path.join(img_dir, str(img_count)) + '.png'
        label = fine_labels[index]
        label = fine_label_names[label]
        imageio.imwrite(filename, image)
        ids.append(img_count)
        labels.append(label)
        img_count += 1

    test = unpickle(data_dir, 'test')

    filenames = [t.decode('utf8') for t in test[b'filenames']]
    fine_labels = test[b'fine_labels']
    data = test[b'data']

    images = list()
    for d in data:
        image = np.zeros((32,32,3), dtype=np.uint8)
        image[...,0] = np.reshape(d[:1024], (32,32)) # Red channel
        image[...,1] = np.reshape(d[1024:2048], (32,32)) # Green channel
        image[...,2] = np.reshape(d[2048:], (32,32)) # Blue channel
        images.append(image)

    for index,image in tqdm(enumerate(images)):
        filename = os.path.join(img_dir, str(img_count)) + '.png'
        label = fine_labels[index]
        label = fine_label_names[label]
        imageio.imwrite(filename, image)
        ids.append(img_count)
        labels.append(label)
        img_count += 1

    df = pd.DataFrame(data = {'id' : ids, 'label' : labels})
    header = ['id', 'label']
    df.to_csv(os.path.join(data_dir, 'cifar100.csv'), columns = header, index=False)

if __name__ == "__main__":
    main()