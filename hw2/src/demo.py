import os 
import numpy as np
from matplotlib import pyplot as plt

from dataset import wafer_dataset as Dataset


LABEL2TEXT = {
    0: 'Central',
    1: 'Donut',
    2: 'Edge-Loc',
    3: 'Edge-Ring',
    4: 'Loc',
    5: 'Near-full',
    6: 'Random',
    7: 'Scratch',
    8: 'None'
}

def display_image_with_gen_data(data, gen_data, label=None):
    """ Display an image and generate samples given their ndarray by matplotlib package.
    Args:
        data: original datas
        gen_data: generated data by encoder
 
    Return:
        None
    """

    N = len(gen_data) # numbers of generated data

    # check shapes of input datas, raise error if it is wrong
    if len(data.shape) != 3 or data.shape[0] != 3:
        raise ValueError('shape of input data should be (3, 26, 26)')

    # check labels of input datas, raise error if it is wrong
    if label is None:
        title = 'No label'
    elif type(label) == int and label in LABEL2TEXT:
        title = LABEL2TEXT[label]
    else:
        raise ValueError('label should be in [0,8]')

    bdry, dfct, nrml = data[0, :, :], data[1, :, :], data[2, :, :]
    ori_img = torch.zeros([26, 26, 3], dtype=torch.float)
    ori_img[:, :, 0] = data[0, :, :]
    ori_img[:, :, 1] = data[1, :, :]
    ori_img[:, :, 2] = data[2, :, :]

    plt.figure()
    plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.05, wspace=0, hspace=0.2)
    plt.suptitle(title, fontsize=14)
    plt.subplot(1+N, 4, 1)
    plt.title('boundary')
    plt.imshow(bdry)
    plt.subplot(1+N, 4, 2)
    plt.title('defect')
    plt.imshow(dfct)
    plt.subplot(1+N, 4, 3)
    plt.title('normal')
    plt.imshow(nrml)
    plt.subplot(1+N, 4, 4)
    plt.title('reconstruct')
    plt.imshow(ori_img)

    for i in range(N):
        img[:, :, 0] = gen_data[i][0, :, :]
        img[:, :, 1] = gen_data[i][1, :, :]
        img[:, :, 2] = gen_data[i][2, :, :]
        gen_bdry = gen_data[i][0, :, :]
        gen_dfct = gen_data[i][1, :, :]
        gen_nrml = gen_data[i][2, :, :]

        plt.subplot(1+N, 4, (i+1)*4 + 1)
        plt.imshow(gen_bdry)
        plt.subplot(1+N, 4, (i+1)*4 + 2)
        plt.imshow(gen_dfct)
        plt.subplot(1+N, 4, (i+1)*4 + 3)
        plt.imshow(gen_nrml)
        plt.subplot(1+N, 4, (i+1)*4 + 4)
        plt.imshow(img)
    
    plt.show()


def demo(data_path, generated_path, demo_index=None):
    """ Display an image in the wafer dataset, and also the corresponding generated samples.

    Args:
        data_path: directory to data
        generated_path: directory to generated data
        demo_index: index of image in the dataset, if None then random generate.
    
    Return:
        None
    """

    dataset = Dataset(data_path)
    gen_dataset = Dataset(generated_path, generated=True)

    if demo_index is None:
        # no index given, just random sample it
        demo_index = np.random.randint(0, len(dataset))
    elif type(demo_index) != int or not(0 <= demo_index < len(dataset)):
        raise ValueError('index should be in [0, {}'.format(len(dataset)))
    
    print('index to demo: {}'.format(demo_index))

    # read original data
    data, label = dataset[demo_index]

    # read generated data
    gen_data = []
    for i in range(5):
        gd, _ = gen_dataset[demo_index * 5 + i]
        gen_data.append(gd)
    
    # display original data and generated data
    display_image_with_gen_data(data, gen_data, int(label))


if __name__ == "__main__":
    demo(data_path='../wafer', generated_path='../output', demo_index=9)
