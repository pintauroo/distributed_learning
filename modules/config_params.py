
import os
import random
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset   
torch.backends.cudnn.benchmark=True

from .VGG_NN import VGG


class ConfigParams:

    ##### Hyperparameters for federated learning #########
    # num_clients = 20
    # num_selected = 6
    # num_rounds = 150
    # epochs = 5
    # batch_size = 32

    def __init__(self, num_clients, num_selected, num_rounds, epochs, batch_size):
        self.num_clients = num_clients
        self.num_selected = num_selected
        self.num_rounds = num_rounds
        self.epochs = epochs
        self.batch_size = batch_size
        self.losses_train = []
        self.losses_test = []
        self.acc_train = []
        self.acc_test = []
   
   

    def set_num_clients(self, value):
        self.num_clients = value

    def get_num_clients(self):
        return self.num_clients
    
    def set_num_selected(self, value):
        self.num_selected = value

    def get_num_selected(self):
        return self.num_selected
    
    def set_num_rounds(self, value):
        self.num_rounds = value

    def get_num_rounds(self):
        return self.num_rounds
    
    def set_epochs(self, value):
        self.epochs = value

    def get_epochs(self):
        return self.epochs
    
    def set_batch_size(self, value):
        self.batch_size = value

    def get_batch_size(self):
        return self.batch_size
    
    def get_opt(self,i):
        return self.opt[i]
    
    def get_losses_train(self):
        return self.losses_train

    def get_losses_test(self):
        return self.losses_test

    def losses_train_append(self, val):
        self.losses_train.append(val)
    
    def losses_test_append(self, test_loss):
        self.losses_test.append(test_loss)

    def acc_test_append(self, acc):
        self.acc_test.append(acc)




