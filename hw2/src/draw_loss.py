import numpy as np
from matplotlib import pyplot as plt


def draw(log_path):
    with open(log_path, 'r') as fp:
        # for _ in range(2):
        #     _ = fp.readline()
        #     print('____', _)
        x, y = [], []
        for i in range(0, 100000, 200):
            line = fp.readline()
            loss = line.split('Train loss')[-1].strip()
            print('i:', i, 'loss:', loss)
            loss = float(loss)
            x.append(i)
            y.append(loss)
    fp.close()

    print('Show figure, len(x) = {}'.format(len(x)))
    print('First loss:{}, last loss:{}'.format(y[0], y[-1]))

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.xticks(np.arange(min(x), max(x)+1000, 20000))
    plt.yticks([max(y), min(y)], [str(max(y)), str(min(y))])
    plt.title('Training Loss vs Epoch')
    plt.plot(x, y)
    plt.show()


if __name__ == "__main__":
    draw('../checkpoints/20211201_171849/log.txt')