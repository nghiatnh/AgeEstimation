import torch
from torch import nn
from matplotlib import pyplot as plt
import cv2
from torch.utils.data import (
    DataLoader, Dataset, BatchSampler, SequentialSampler,)
import pandas as pd
import copy
from tqdm import tqdm

device = "cuda"
torch.manual_seed(42)


def visualize(image):
  plt.figure(figsize=(7, 7))
  plt.axis("off")
  plt.imshow(image)


class ImageAgeDataset(Dataset):
  def __init__(self, images: pd.DataFrame):
    self.group = images.Group.values
    self.paths = images.Filepath.values

  def __len__(self):
    return len(self.group)

  def __getitem__(self, item):
    out = dict()
    path = self.paths[item]
    image = cv2.imread(path)
    image = cv2.resize(image, (128, 128))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    out['x'] = torch.tensor(image, dtype=torch.float).permute(2, 0, 1)/255
    out["y"] = torch.tensor(self.group[item], dtype=torch.long)
    return out


class ImageLoader(DataLoader):
  def __init__(self, images: pd.DataFrame, batch_size: int = 50,
               drop_last: bool = False, **data_loader_kwargs):
    self.dataset = ImageAgeDataset(images)
    self.data_loader_kwargs = copy.deepcopy(data_loader_kwargs)
    sampler = BatchSampler(
        SequentialSampler(self.dataset),
        batch_size=batch_size, drop_last=drop_last)
    self.data_loader_kwargs.update({"sampler": sampler, "batch_size": None})
    super().__init__(
        self.dataset, **data_loader_kwargs)


class AlexNetwork(nn.Module):
  def __init__(self, n_classes):
    super(AlexNetwork, self).__init__()
    self.n_classes = n_classes
    self.conv_1 = nn.Conv2d(
        in_channels=3,
        out_channels=96,
        kernel_size=11,
        stride=4,
        padding=50,
    )
    self.pool_1 = nn.MaxPool2d(
        kernel_size=3,
        stride=2,
    )
    self.conv_2 = nn.Conv2d(
        in_channels=96,
        out_channels=256,
        kernel_size=5,
        stride=1,
        padding=2
    )
    self.pool_2 = nn.MaxPool2d(
        kernel_size=3,
        stride=2,
    )
    self.conv_3 = nn.Conv2d(
        in_channels=256,
        out_channels=384,
        kernel_size=3,
        stride=1,
        padding=1
    )
    self.conv_4 = nn.Conv2d(
        in_channels=384,
        out_channels=384,
        kernel_size=3,
        stride=1,
        padding=1
    )
    self.conv_5 = nn.Conv2d(
        in_channels=384,
        out_channels=256,
        kernel_size=3,
        stride=1,
        padding=1
    )
    self.pool_3 = nn.MaxPool2d(
        kernel_size=3,
        stride=2,
    )
    self.nn = nn.Sequential(
        nn.Linear(in_features=9216, out_features=4096),
        nn.ReLU(),
        nn.Linear(in_features=4096, out_features=4096),
        nn.ReLU(),
        nn.Linear(in_features=4096, out_features=self.n_classes),)

  def forward(self, x):
    x = self.conv_1(x)
    x = self.pool_1(x)
    x = self.conv_2(x)
    x = self.pool_2(x)
    x = self.conv_3(x)
    x = self.conv_4(x)
    x = self.conv_5(x)
    x = self.pool_3(x)
    x = x.view(-1, 9216)
    x = self.nn(x)
    return x


loss_fn = nn.CrossEntropyLoss()
activate = nn.Softmax(dim=-1)


def train(model, optimizer, dataloader):
  model.train()
  total_loss = 0
  count = 0
  optimizer.zero_grad()
  for data in tqdm(dataloader):
    x = data["x"].to(device)
    y = data["y"].to(device)
    y_hat = model(x).to(device)
    loss = loss_fn(y_hat, y)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
    if torch.argmax(activate(y_hat)).item() == y.item():
      count += 1
  return total_loss/len(dataloader), count / len(dataloader)


def valid(model, dataloader):
  model.eval()
  total_loss = 0
  count = 0
  for data in tqdm(dataloader):
    x = data["x"].to(device)
    y = data["y"].to(device)
    with torch.no_grad():
      y_hat = model(x).to(device)
      loss = loss_fn(y_hat, y)
      if torch.argmax(activate(y_hat)).item() == y.item():
        count += 1
    total_loss += loss.item()
  return total_loss/len(dataloader), count / len(dataloader)


def pred(x):
  return torch.argmax(x)
