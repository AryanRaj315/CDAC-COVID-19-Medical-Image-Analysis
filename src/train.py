import os
import cv2
import glob
from math import atan2, asin
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from torch.utils.data import DataLoader, Dataset, sampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from albumentations.pytorch import ToTensor
import torch
from torchvision import transforms
import torch.nn as nn
import time

from torch.nn import functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.models as models
try:
    from ralamb import Ralamb
    from radam import RAdam
    from ranger import Ranger
    from lookahead import LookaheadAdam
    from over9000 import Over9000
    from tqdm.notebook import tqdm
except:
    os.system(f"""git clone https://github.com/mgrankin/over9000.git""")
    import sys
    sys.path.insert(0, 'over9000/')
    from ralamb import Ralamb
    from radam import RAdam
    from ranger import Ranger
    from lookahead import LookaheadAdam
    from over9000 import Over9000
    from tqdm.notebook import tqdm

from meter import *
from utils import *
from dataset import *

class Trainer(object):
    """
        This class takes care of training and validation of our model
    """
    def __init__(self,model, optim, lr, bs, epochs = 20, name = 'model', shape=200):
        self.batch_size = bs
        self.accumulation_steps = 1
        self.lr = lr
        self.name = name
        self.num_epochs = epochs
        self.optim = optim
        self.best_loss = float("inf")
        self.phases = ["train", "val"]
        self.device = torch.device("cuda:0")
        torch.set_default_tensor_type("torch.FloatTensor")
        self.net = model
        self.best_val_acc = 0
        self.best_val_loss = 10
        self.best_f1_score = 0
        self.losses = {phase: [] for phase in self.phases}
        self.criterion = BCEDiceLoss()
        if self.optim == 'Over9000':
            self.optimizer = Over9000(self.net.parameters(),lr=self.lr)
        elif self.optim == 'Adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(),lr=self.lr)
        elif self.optim == 'RAdam':
            self.optimizer = Radam(self.net.parameters(),lr=self.lr)
        elif self.optim == 'Ralamb':
            self.optimizer = Ralamb(self.net.parameters(),lr=self.lr)
        elif self.optim == 'Ranger':
            self.optimizer = Ranger(self.net.parameters(),lr=self.lr)
        elif self.optim == 'LookaheadAdam':
            self.optimizer = LookaheadAdam(self.net.parameters(),lr=self.lr)
        else:
            raise(Exception(f'{self.optim} is not recognized. Please provide a valid optimizer function.'))
            
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=3, verbose=True,factor = 0.5,min_lr = 1e-5)
        self.net = self.net.to(self.device)
        cudnn.benchmark = True
        self.dataloaders = {
            phase: provider(
                phase=phase,
                batch_size=self.batch_size
            )
            for phase in self.phases
        }
        self.losses = {phase: [] for phase in self.phases}
        self.acc_scores = {phase: [] for phase in self.phases}
        self.f1_scores = {phase: [] for phase in self.phases}
    
        def load_model(self, name, path='models/'):
            state = torch.load(path+name, map_location=lambda storage, loc: storage)
            self.net.load_state_dict(state['state_dict'])
            self.optimizer.load_state_dict(state['optimizer'])
            print("Loaded model with dice: ", state['best_acc'])
    
    def forward(self, images, targets):
        images = images.to(self.device)
        targets = targets.type("torch.FloatTensor")
        targets = targets.to(self.device)
        preds = self.net(images)
        preds.to(self.device)
        loss = self.criterion(preds,targets)
        # Calculating accuracy of the predictions
#         probs = torch.sigmoid(preds)
#         probs_cls = torch.sigmoid(preds)
#         acc = accuracy_score(probs_cls.detach().cpu().round(), targets.detach().cpu())
        return loss, preds

    def iterate(self, epoch, phase):
        meter = Meter(phase, epoch)
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase: {phase} | ⏰: {start}")
        batch_size = self.batch_size
        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)
        tk0 = tqdm(dataloader, total=total_batches)
        self.optimizer.zero_grad()
        for itr, batch in enumerate(tk0): 
            images, targets = batch
            loss, preds= self.forward(images, targets)
            loss = loss / self.accumulation_steps
            if phase == "train":
                loss.backward()
                if (itr + 1 ) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
            preds = preds.detach().cpu()
            targets = targets.detach().cpu()
            meter.update(targets, preds)
            tk0.set_postfix(loss=(running_loss / ((itr + 1))))
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        acc, f1, precision, recall = epoch_log(phase, epoch, epoch_loss, meter, start)
        self.losses[phase].append(epoch_loss)
        self.acc_scores[phase].append(acc)
        torch.cuda.empty_cache()
        return epoch_loss, acc, f1, precision, recall

    def train_end(self):
        train_loss = self.losses["train"]
        val_loss = self.losses["val"]
        train_acc = self.acc_scores["train"]
        val_acc = self.acc_scores["val"]
        df_data=np.array([train_loss, train_acc, val_loss, val_acc]).T
        df = pd.DataFrame(df_data,columns = ['train_loss','train_acc', 'val_loss', 'val_acc'])
        df.to_csv('logs/'+self.name+".csv")
        
    def predict(self):
        self.net.eval()
        with torch.no_grad():
            self.iterate(1,'test')
        print('Done')

    def fit(self, epochs):
#         self.num_epochs+=epochs
        for epoch in range(0, self.num_epochs):
            self.iterate(epoch, "train")
            state = {
                "epoch": epoch,
                "best_loss": self.best_val_loss,
                "best_f1": self.best_f1_score,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            self.net.eval()
            with torch.no_grad():
                epoch_loss, acc, f1, precision, recall = self.iterate(epoch, "val")
                self.scheduler.step(epoch_loss)
            if f1 >  self.best_f1_score:
                print("* New optimal found according to f1 score, saving state *")
                state["best_f1"] = self.best_f1_score = f1
                os.makedirs('models/', exist_ok=True)
                torch.save(state, 'models/'+self.name+'_best_f1.pth')
            if epoch_loss <  self.best_val_loss:
                print("* New optimal found according to val loss, saving state *")
                state["best_loss"] = self.best_val_loss = epoch_loss
                os.makedirs('models/', exist_ok=True)
                torch.save(state, 'models/'+self.name+'_best_loss.pth')
            print()
            self.train_end()