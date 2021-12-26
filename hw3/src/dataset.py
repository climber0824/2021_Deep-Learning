import os
import cv2
import numpy as np

class Dataset:

    img_h = 32
    img_w = 32
    img_c = 1
    text2label = {'Carambula': 0, 'Lychee': 1, 'Pear':2}
    
    def __init__(self, data_path, mode):
        
        assert mode in ['train', 'val', 'test'], 'mode should be train, val or test'
        
        self.mode = mode
        self.images = []
        self.labels = []

        if self.mode == 'train':
            
            self.data_path = os.path.join(data_path, 'Data_train')
            
            for text, label in self.text2label.items():
                for filename in os.listdir(os.path.join(self.data_path, text))[:343]:
                    img = cv2.imread(os.path.join(self.data_path, text, filename), 0)
                    self.images.append(img)
                    self.labels.append(label)

        elif self.mode == 'val':
            self.data_path = os.path.join(data_path, 'Data_train')
            for text, label in self.text2label.items():
                for filename in os.listdir(os.path.join(self.data_path, text))[343:]:
                    img = cv2.imread(os.path.join(self.data_path, text, filename), 0)
                    self.images.append(img)
                    self.labels.append(label)
                
        else:
            self.data_path = os.path.join(data_path, 'Data_test')
            for text, label in self.text2label.items():
                for filename in os.listdir(os.path.join(self.data_path, text)):
                    img = cv2.imread(os.path.join(self.data_path, text, filename), 0)
                    self.images.append(img)
                    self.labels.append(label)


    def __len__(self):
        return len(self.images)


    def __getitem__(self, int_or_slice):
        if isinstance(int_or_slice, int):
            indexes = [int_or_slice]
        elif isinstance(int_or_slice, slice):
            start = int_or_slice.start if int_or_slice.start else 0
            stop = int_or_slice.stop if int_or_slice.stop else len(self.images)
            step = int_or_slice.step if int_or_slice.step else 1
            indexes = list(iter(range(start, stop, step)))
        else:
            indexes = list(int_or_slice)

        num_of_fetch = len(indexes)
        images, labels = [], []
        for i, idx in enumerate(indexes):
            image = self.images[idx]
            image = np.array(image, dtype=float)
            image /= 255.0
            images.append(image)

            label = self.labels[idx]
            labels.append(label)

        images = np.array(images, dtype=float)
        images = np.reshape(images, (num_of_fetch, self.img_h, self.img_w, self.img_c))
        labels = np.array(labels, dtype=int)
        labels = np.reshape(labels, (num_of_fetch, 1))

        return images, labels


if __name__ == "__main__":
    dataset = Dataset("../Data", 'train')
    print(dataset[0])
