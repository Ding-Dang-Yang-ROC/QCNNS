import numpy as np
import torch

N_fn=8
S_Ker=16*16
N_par=S_Ker*S_Ker
N_Ite=1000
learning_rate=2.0 # < 3.0
Batch=100

Dtype=torch.float64
dev2="cpu" # "cuda"


