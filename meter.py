import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, recall_score, precision_score
import torch

class Meter:
    '''A meter to keep track of iou and dice scores throughout an epoch'''
    def __init__(self, phase, epoch):
        self.acc_scores = []
        self.f1_scores = []
        self.precision_scores = []
        self.recall_scores = []
        self.phase = phase

    def update(self, targets, outputs):
        probs = torch.sigmoid(outputs)
        probs_cls = torch.sigmoid(outputs)
        precision = precision_score(targets, probs_cls.argmax(axis=1), average = 'micro', labels = [0,1])
        recall = recall_score(targets, probs_cls.round(), average = 'micro', labels = [0,1])
        f1 = f1_score(targets, probs_cls.round(), average='micro', labels = [0,1])
        acc = accuracy_score(targets, probs_cls.round())
        # Adding all metrics to list
        self.acc_scores.append(acc)
        self.f1_scores.append(f1)
        self.precision_scores.append(precision)
        self.recall_scores.append(recall)

    def get_metrics(self):
        acc = np.nanmean(self.acc_scores)
        f1 = np.nanmean(self.f1_scores)
        precision = np.nanmean(self.precision_scores)
        recall = np.nanmean(self.recall_scores)
        return acc, f1, precision, recall
    
def epoch_log(phase, epoch, epoch_loss, meter, start):
    '''logging the metrics at the end of an epoch'''
    acc, f1, precision, recall = meter.get_metrics()
    print("Loss: %0.4f | accuracy: %0.4f | F1: %0.4f | Precision: %0.4f | Recall: %0.4f" % (epoch_loss, acc, f1, precision, recall))
    return acc, f1, precision, recall