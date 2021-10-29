import os
import numpy as np
import cv2
import glob

class MNIST_dataset():

    def __init__(self, data_path, mode):
        assert mode in ["train", "val", "test"], "mode should be train, val, or test"
        self.mode = mode
        self.image_path = os.path.join(data_path, "img")
        self.label_path = os.path.join(data_path, "label")

        self.images = []
        self.labels = []

        if self.mode == "train":
            for filename in sorted(glob.glob(os.path.join(self.image_path, "*.png")))[:42000]:
                img = cv2.imread(filename, 0)
                self.images.append(img)
            
            for filename in sorted(glob.glob(os.path.join(self.label_path, "*.txt")))[:42000]:
                with open(filename, "r") as f:
                    label = f.readline()
                    self.labels.append(label)
                
                f.close()

        elif self.mode == "val":
            for filename in sorted(glob.glob(os.path.join(self.image_path, "*.png")))[42000:]:
                img = cv2.imread(filename, 0)
                self.images.append(img)
            
            for filename in sorted(glob.glob(os.path.join(self.label_path, "*.txt")))[42000:]:
                with open(filename, "r") as f:
                    label = f.readline()
                    self.labels.append(label)
                
                f.close()
        
        elif self.mode == "test":
            for filename in sorted(glob.glob(os.path.join(self.image_path, "*.png"))):
                img = cv2.imread(filename, 0)
                self.images.append(img)
            
            for filename in sorted(glob.glob(os.path.join(self.label_path, "*.txt"))):
                with open(filename, "r") as f:
                    label = f.readline()
                    self.labels.append(label)
                
                f.close()


    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        image = self.images[index]
        image = np.array(image, dtype=float)
        image = image.reshape(1, -1)
        image /= 255.0

        label = self.labels[index]
        label = np.array(label, dtype=int)
        label = label.reshape(1, -1)

        return {"image": image, "label": label}



if __name__ == "__main__":
    data_path = "./MNIST/train"
    dataset = MNIST_dataset(data_path, "train")
    data = dataset[0]
    img, label = data["image"], data["label"]
    
