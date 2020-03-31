import time
import datetime
import numpy as np
import torch
from scipy import sparse
import torch.nn.functional as F
from trans_graph import *
from matplotlib import pyplot as plt

a = torch.tensor([1,2,3,5,4,3,9,0,1])
q = torch.topk(a,k=3)[1]
print(q)
for i in q:
    print(i//3)
    print(i%3)