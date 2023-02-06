import os
import io
from PIL import Image
from torchvision.transforms import ToTensor
import argparse
import json
import logging
import time
import random
import re
import sys
import pdb
import datetime
from geomdl import ray

import attr
import torch
import numpy as np
import tqdm
import torch.utils.data as data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import cv2



class sequenceDataset(data.Dataset):

    def __init__(self, file, len_past, num_coords):
        self.name = 'toy_social'
        self.len_past = len_past
        self.num_coords = num_coords
        self.data = np.load(file, allow_pickle=True)

    # def __getitem__(self, idx):
    #     sample = torch.Tensor(self.data[idx])
    #     past = sample[:, :self.len_past]
    #     future = sample[:, self.len_past:]
    #     num_agents = past.shape[0]
    #
    #     return past, future, num_agents

    def __getitem__(self, idx):
        sample = self.data[idx]

        # shuffle agents
        # idx = torch.randperm(sample.shape[0])
        # track = sample[idx]

        # rotation agents
        # angle = random.uniform(0, 360)
        # matRot_track = cv2.getRotationMatrix2D((0, 0), angle, 1)
        # temp = []
        # track = np.array(track)
        # for i in range(track.shape[0]):
        #     temp.append(cv2.transform(track[i].reshape(-1, 1, 2), matRot_track).squeeze())
        # track = np.array(temp)s


        track = torch.Tensor(sample)

        #spost = random.uniform(0.0, 5.0)
        past = track[:, :self.len_past]# + spost
        future = track[:, self.len_past:]# + spost
        num_agents = track.shape[0]

        return past, future, num_agents

    def __len__(self):
        return len(self.data)
