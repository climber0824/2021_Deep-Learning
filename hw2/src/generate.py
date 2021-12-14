import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from model import autoEncoder as AE
from dataset import wafer_dataset as Dataset

def generate(data_path, generated_path, checkpoint_path):

    dataset = Dataset(data_path=data_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = AE()
    model.load(checkpoint_path)
    model = model.float()
    model = model.cuda()
    
    generated_img = np.zeros((1, 26, 26, 3))
    generated_label = np.zeros((1, 1))

    save_ = True

    for batch_idx, (data, label) in enumerate(dataloader):
        data = data.float()
        data = data.cuda()

        latent = model.eval().encode(data)

        for _ in range(5):
            noise = torch.randn_like(latent)
            noise_latent = latent.detach().clone() + noise

            reconstructed_data = model.eval().decode(noise_latent)
            reconstructed_data = reconstructed_data.detach().cpu()
            reconstructed_data = reconstructed_data.numpy()
            reconstructed_data = np.transpose(reconstructed_data, (0,2,3,1))
            
            for i in range(26):
                for j in range(26):
                    max_channel = np.argmax(reconstructed_data[0, i, j, :])
                    for c in range(3):
                        reconstructed_data[0, i, j, c] = 1 if c == max_channel else 0

            generated_img = np.concatenate((generated_img, reconstructed_data), axis=0)
            generated_label = np.concatenate((generated_label, label), axis=0)

    generated_img = generated_img[1:]
    generated_label = generated_label[1:]

    if save_:
        np.save(os.path.join(data_path, 'gen_data'), generated_img)
        np.save(os.path.join(data_path, 'gen_label'), generated_label)

if __name__ == "__main__":
    #generate(data_path="../wafer", generated_path="../output", checkpoint_path="../checkpoints/model_last.pth")
    gen_img = np.load("../wafer/gen_data.npy")
    print(gen_img[0].shape())