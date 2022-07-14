import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm


from .utilityNet import I3Du
from .budgetNet import I3Db
from .degradNet import resnet_degrad





def i3d(num_classes, dropout, without_t_stride, pooling_method, **kwargs):

    model_degrad =  resnet_degrad()
    model_utility = I3Du()
    model_budget = I3Db()

    model_utility1 = I3Du()
    model_budget1 = I3Db()



    return [model_degrad, model_utility, model_budget, model_utility1, model_budget1] 







