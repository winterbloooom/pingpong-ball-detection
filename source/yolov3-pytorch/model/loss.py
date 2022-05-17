import torch
import torch.nn as nn
import numpy as np

def celoss(input, target):
    loss = nn.CrossEntropyLoss()
    output = loss(input, target)
    #print("input : {}, target : {}, output : {}".format(input, target, output))
    
def bceloss(input, target):
    loss = nn.BCEWithLogitsLoss()
    output = loss(input, target)