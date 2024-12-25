import torch
import torch.nn as nn
import requests
from PIL import Image
import numpy as np
import io
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import os


def load_image_from_url(url, size=64):
    response = requests.get(url)
    img = Image.open(io.BytesIO(response.content))
    img = img.resize((size, size))
    img = img.convert("RGBA")
    img_tensor = torch.from_numpy(np.array(img))
    return img_tensor / 255


def damage(image, radius):
    height, width = image.shape[:2]
    x_center = random.randint(0, width)
    y_center = random.randint(0, height)
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    mask = (x - x_center)**2 + (y - y_center)**2 <= radius**2
    image[mask, ..., :3] = 1


class NeuralCellularAutomata(nn.Module):
    def __init__(self, device=None):
        super(NeuralCellularAutomata, self).__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        sobel_x = torch.tensor([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]], dtype=torch.float) / 8

        # Initialize Sobel filters for edge detection
        self.conv_sobel_x = nn.Conv2d(16, 16, 3, padding='same')
        self.conv_sobel_x.weight = nn.Parameter(
            torch.tile(sobel_x, (16, 16, 1, 1)),
            requires_grad=False
        )

        self.conv_sobel_y = nn.Conv2d(16, 16, 3, padding='same')
        self.conv_sobel_y.weight = nn.Parameter(
            torch.tile(sobel_x.T, (16, 16, 1, 1)),
            requires_grad=False
        )

        # Define fully connected layers
        self.dense1 = nn.Linear(48, 128)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(128, 16)
        self.dense2.weight.data.zero_()

        # Define loss function and optimizer
        self.loss_fn = nn.MSELoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.5)
        self.to(self.device)

    def save(self, path='model.pth'):
        torch.save(self.state_dict(), path)

    def load(self, path='model.pth'):
        self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    def stochastic_update(self, x, dx, fire_rate=0.5):
        """
        x: (batch_size, channels, height, width)
        dx: (batch_size, height, width, channels)
        out_x : (batch_size, height, width, channels)
        """
        # Apply stochastic update to the state grid
        rand_mask = (torch.rand((dx.size(0), dx.size(1), dx.size(2), 1)) < fire_rate).to(self.device).float()
        dx = dx * rand_mask
        return x + dx.transpose(1, 3)

    def perceive(self, x):
        """
        x: (batch_size, channels, height, width)
        out_x : (batch_size, channels, height, width)
        """
        # Forward pass through the network
        x = torch.cat((self.conv_sobel_x(x), self.conv_sobel_y(x), x), 1)
        return x.transpose(1, 3)

    def forward(self, x):
        """
        x: (batch_size, channels, height, width)
        out_x : (batch_size, channels, height, width)
        """
        # Forward pass through the network
        x = self.perceive(x)
        x = self.relu(self.dense1(x))
        x = self.dense2(x)
        return x

    def update(self, x):
        """
        x: (batch_size, channels, height, width)
        out_x : (batch_size, channels, height, width)
        """
        # Update the state grid
        dx = self(x * self._get_living_mask(x))
        x = self.stochastic_update(x, dx)
        return x
    
    def train(self, batch, target):
        """
        batch: (batch_size, channels, height, width)
        target: (batch_size, channels, height, width)
        outputs: (batch_size, channels, height, width)
        """
        # Perform optimization step
        self.optimizer.zero_grad()
        for _ in range(40):
            outputs = self.update(batch)
        loss = self.loss_fn(outputs[:, :4, :, :], target[:, :4, :, :])
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return outputs, loss
    
    def _get_living_mask(self, x):
        mask = torch.nn.functional.max_pool2d(x[:, 3:4, :, :], kernel_size=3, stride=1, padding=1) > 0.1
        return mask.float()

    def pool_training(self, target, epochs=50000, pool_size=1024, batch_size=25):
        """
        target: (channels, height, width)
        """
        # Initialize the seed and pool
        channels, height, width = target.shape
        seed = torch.zeros(16, height, width)
        seed[3:, height // 2, width // 2] = 1
        targets = target.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(self.device)
        pool = [seed.clone() for _ in range(pool_size)]
        
        def sample():
            return [list(x) for x in zip(*random.sample(list(enumerate(pool)), batch_size))]

        for i in tqdm(range(epochs)):
            idxs, batch = sample()
            with torch.no_grad():
                # Sort batch by loss
                sorted_idx = sorted(range(batch_size),
                            key=lambda x: self.loss_fn(
                            batch[x][:4, :, :].unsqueeze(0).to(self.device),
                            targets[x].unsqueeze(0).clamp(0, 1).to(self.device)),
                            reverse=True
                            )
            idxs = [idxs[i] for i in sorted_idx]
            batch = [batch[i] for i in sorted_idx]
            batch[0] = seed.clone()

            # Apply damage to a portion of the batch
            for j in range(batch_size - int(round(batch_size * 0.2)), batch_size):
                damage(batch[j], random.randint(batch[j].shape[1] // 12, batch[j].shape[1] // 5))
            
            # Perform optimization step
            outputs, loss = self.train(torch.stack(batch).to(self.device), targets)
            print(f"Loss: {loss}")
            outputs = outputs.cpu().detach()
            for idx, output in zip(idxs, outputs):
                pool[idx] = output.cpu().detach()
                del output


if __name__ == "__main__":
    nca = NeuralCellularAutomata()
    image_url = "https://static.vecteezy.com/system/resources/previews/003/240/508/original/beautiful-purple-daisy-flower-isolated-on-white-background-vector.jpg"
    nca.pool_training(load_image_from_url(image_url).transpose(0, 2), epochs=100, pool_size=50, batch_size=10)
    # nca.save()