import datetime
import torch.optim as optim
import os
from torch.utils.data import DataLoader

from dataset import wafer_dataset as Dataset
from model import autoEncoder as AE

def train(data_path, checkpoint_path):
    """ Training process
    Args:
        data_path: path to training datas 
        checkpoint_path: path to checkpoints
    
    Return:
        None
    """

    time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.join(checkpoint_path, time), exist_ok=True)

    dataset = Dataset(data_path=data_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # create model
    model = AE()
    model = model.float()
    model = model.cuda()

    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epoch_num = 100000
    for epoch_idx in range(1, epoch_num+1):
        total_loss = []
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.float()
            data = data.cuda()

            # encode the data to latent
            latent = model.train().encode(data)

            # decode the latent to reconstructed img
            reconstructed_data = model.train().decode(latent)

            # compute loss
            loss = model.loss(data, reconstructed_data)

            total_loss.append(loss)
            optimizer.zero_grad()

            # backward
            loss.backward()

            # update model weight
            optimizer.step()

        # print training informations
        if epoch_idx % 200 == 0:
            print("[{}] Epoch: {:2d} | Loss: {:.4f}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch_idx, sum(total_loss) / len(total_loss)))
             # write to log file
            with open(os.path.join(checkpoint_path, time, "log.txt"), "a") as fp:
                fp.write("[{}] Epoch {:2d} | Train loss {:.4f}\n".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch_idx, sum(total_loss) / len(total_loss)))
            fp.close()
        
        if epoch_idx % 2000 == 0:
            # save current model weight
            model.save(checkpoint_path, tag=str(epoch_idx))
    
    model.save(checkpoint_path, tag='last')

if __name__ == "__main__":
    train(data_path="../wafer", checkpoint_path="../checkpoints")