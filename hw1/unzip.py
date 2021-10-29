import gzip
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt


def training_images(src_path, dst_path):

    with gzip.open(src_path, 'r') as f:
        
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        
        # second 4 bytes is the number of images
        image_count = int.from_bytes(f.read(4), 'big')
        
        # third 4 bytes is the row count
        row_count = int.from_bytes(f.read(4), 'big')
        
        # fourth 4 bytes is the column count
        column_count = int.from_bytes(f.read(4), 'big')
        
        # rest is the image pixel data, each pixel is stored as an unsigned byte
        
        # pixel values are 0 to 255
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8)\
            .reshape((image_count, row_count, column_count))
        
    f.close()
    if os.path.isdir(dst_path):
        for i in range(image_count):
            cv2.imwrite(os.path.join(dst_path, str(i).zfill(5)+ ".png"), images[i])
    else:
        print('Path does not exist')
    

def training_labels(src_path, dst_path):
    with gzip.open(src_path, 'r') as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes is the number of labels
        label_count = int.from_bytes(f.read(4), 'big')
        # rest is the label data, each label is stored as unsigned byte
        # label values are 0 to 9
        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)
    
    f.close()
    if os.path.isdir(dst_path):
        for i in range(label_count):
            with open(os.path.join(dst_path, str(i).zfill(5) + ".txt"), "w") as fp:
                fp.write(str(labels[i]))
        fp.close()
    else:
        print('Path does not exist')


if __name__ == "__main__":
    img_src_path = '/home/kenchang/projects/DL/hw1/MNIST/train-images-idx3-ubyte.gz'
    img_dst_path = '/home/kenchang/projects/DL/hw1/MNISTtrain/img/'
    label_src_path = '/home/kenchang/projects/DL/hw1/MNIST/train-labels-idx1-ubyte.gz'
    label_dst_path = '/home/kenchang/projects/DL/hw1/MNIST/train/label/'
    training_images(src_path, dst_path)
    training_labels(label_src_path, label_dst_path)
