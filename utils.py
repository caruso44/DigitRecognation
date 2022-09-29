import pandas as pd
import numpy as np
from torchvision import transforms
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import MNIST

def read_data():
    BATCH_SIZE = 100
    VALID_SIZE = 0.15
    transform_train = transforms.Compose([
    transforms.ToPILImage(),
   # transforms.RandomRotation(0, 0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    transform_valid = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    dataset = pd.read_csv("Dataset/train.csv")

        # Creating datasets for training and validation
    train_data = MNIST.DatasetMNIST(dataset, transform=transform_train)
    valid_data = MNIST.DatasetMNIST(dataset, transform=transform_valid)

    # Shuffling data and choosing data that will be used for training and validation
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(VALID_SIZE * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=BATCH_SIZE, sampler=valid_sampler)
    return train_loader, valid_loader




