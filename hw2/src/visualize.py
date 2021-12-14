from dataset import wafer_dataset as Dataset

from matplotlib import pyplot as plt

label_to_text = {
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

def show_img(data, label=None):

    if len(data.shape) != 3 or data.shape[0] != 3:
        raise ValueError('Shape of input datas should be (3, 26, 26)')
    
    if label is None:
        title = 'No label provided'
    elif type(label) == int and label in label_to_text:
        title = label_to_text[label]
    else:
        raise ValueError('Label is wrong!')
    
    bdry, dfct, nrml = data[0, :, :], data[1, :, :], data[2, :, :]

    plt.figure()
    plt.suptitle(title, fontsize=14)
    plt.subplot(1, 3, 1)
    plt.title('boundary')
    plt.imshow(bdry)

    plt.subplot(1, 3, 2)
    plt.title('defect')
    plt.imshow(dfct)

    plt.subplot(1, 3, 3)
    plt.title('normal')
    plt.imshow(nrml)

    plt.show()


if __name__ == "__main__":
    dataset = Dataset(data_path='../wafer', generated=True)
    print(len(dataset))
    for i in range(0, len(dataset)):
        data, label = dataset[i]
        if label != 8:
            continue
        print(i)
        show_img(data, int(label))
        break