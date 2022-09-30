import torch

a=torch.rand(4,2)
print(a,"\n")
print(a.shape,"\n")
print(a.min(axis=-1))