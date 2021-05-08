#%%
import torch.nn as nn
import torch
#%%
if __name__ == "__main__":
    #%%
    loss = nn.MSELoss(reduction='mean')
    input = torch.tensor([0.5, 0.5]).view(1, -1)
    target = torch.tensor([0.3, 0.4]).view(1, -1)
    output = loss(input, target)
    print(output)