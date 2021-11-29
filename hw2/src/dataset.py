import os
import numpy as np
import torch.utils.data

from torchvision.transforms import transforms

class wafer_dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, generated=False):
        """ Wafer dataset
        """

        if generated:
            self.datas = np.load(os.path.join(data_path, 'gen_data.npy'))
            self.labels = np.load(os.path.join(data_path, 'gen_label.npy'))
        else:
            self.datas = np.load(os.path.join(data_path, 'data.npy'))
            self.labels = np.load(os.path.join(data_path, 'label.npy'))
    
    def __len__(self):
        """ Return the length of dataset
        """
        l = self.datas.shape[0]

        return l

    def __getitem__(self, index):
        """ Get item from dataset given an index
        """

        data = self.datas[index]
        label = self.labels[index]

        data = self.preprocess(data)
        data.view(3, 26, 26)
        label = label.reshape(1)

        return data, label

    @staticmethod
    def preprocess(data):
        """ Convert data into tensor
        """

        data = transforms.ToTensor()(data)

        return data


if __name__ == "__main__":
    dataset = wafer_dataset(data_path='../wafer', generated=False)
    i, l = dataset[0]
    print(type(i), i.shape)