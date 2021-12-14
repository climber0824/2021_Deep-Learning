import os
import torch
import numpy as np

from torch import nn

class autoEncoder(nn.Module):
    def __init__(self):
        """ An AutoEncoder for generating wafer images
        """
        super(autoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, padding=0),
            nn.ReLU()
        ) 

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=5, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=7, padding=0),
            nn.ReLU()
        )

    def encode(self, data):
        """
        Args:
            data: shape = (batch_size, 3, 26, 26)
        
        Return:
            latent: output encoded latent, shape = (batch_size, 8, 12, 12)
        """

        latent = self.encoder(data)

        return latent

    def decode(self, latent):
        """
        Args:
            latent: shape = (batch_size, 8, 12, 12)
        
        Return:
            output: shape = (batch_size, 3, 26, 26)
        """

        output = self.decoder(latent)

        return output
    
    def loss(self, data, reconstructed_data):
        """
        Args:
            data: input images, shape = (batch_size, 3, 26, 26)
            reconstructed_data: images after decoder, shape = (batch_size, 3, 26, 26)
        
        Return:
            loss.shape = ()
        """

        loss_func = nn.MSELoss()
        loss = loss_func(data, reconstructed_data)
        
        return loss

    def save(self, checkpoints_path, tag):
        """
        Args:
            checkpoints_path: directory to save model weights
            tag: the current model weight
        
        Return:
            None
        """
        filename = os.path.join(checkpoints_path, "model_{}.pth".format(str(tag)))
        torch.save(self.state_dict(), filename)

    def load(self, checkpoints_path):
        """
        Args:
            checkpoints_path: path to wanted model weight
        
        Return:
            None
        """
        
        self.load_state_dict(torch.load(checkpoints_path))   