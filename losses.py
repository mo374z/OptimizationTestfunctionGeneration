import torch
from torch import nn
import torch.nn.functional as F
import fastdtw


# Cosine Loss
class CosineLoss(nn.Module):
    def __init__(self, cosine_weight=1.0, mse_weight=1.0):
        super(CosineLoss, self).__init__()
        self.cosine_weight = cosine_weight
        self.mse_weight = mse_weight

    def forward(self, predictions, targets):
        # Reshape the tensors to have at least 2 dimensions
        predictions = predictions.unsqueeze(1)
        targets = targets.unsqueeze(1)
        
        cosine_loss = 1.0 - F.cosine_similarity(predictions, targets, dim=2).mean()
        mse_loss = nn.MSELoss()(predictions, targets)

        total_loss = self.cosine_weight * cosine_loss + self.mse_weight * mse_loss
        return total_loss
    

# Custom Loss
class CustomLoss(torch.nn.Module):
    def __init__(self, global_weight=1.0, local_weight=1.0):
        super(CustomLoss, self).__init__()
        self.global_weight = global_weight
        self.local_weight = local_weight

    def forward(self, predictions, targets):
        global_loss = torch.mean((predictions - targets) ** 2)
        local_loss = torch.mean(torch.exp((predictions - targets) ** 2) - 1)

        total_loss = self.global_weight * global_loss + self.local_weight * local_loss
        return total_loss

# SoftDTW Loss
def soft_dtw(y_pred, y_true, gamma=1.0):
    # Calculate the Soft-DTW loss
    d_mat = torch.cdist(y_pred.unsqueeze(1), y_true.unsqueeze(1), p=2)
    D = torch.zeros_like(d_mat)
    
    for i in range(D.shape[1]):
        for j in range(D.shape[2]):
            D[:, i, j] = torch.sqrt(torch.clamp(d_mat[:, i, j] ** 2, min=1e-10))
    
    D = D.cumsum(dim=1)
    D = D.cumsum(dim=2)
    
    return D[:, -1, -1]


# custom loss for derivatives of order n
def higher_order_derivatives(f, wrt, n):
    derivatives = [f.sum()]
    for f_ in f:    
        print("juan")
        for _ in range(n):
            print("karlo")

            grads = torch.autograd.grad(f_, wrt, create_graph=True)[0]
            derivatives.append(grads.sum())
            f = grads.sum()
    
    return torch.stack(derivatives)


class KnollHaZeHei(torch.nn.Module):
    def __init__(self, diff_degree, criterion):
        super().__init__()
        self.diff_degree = diff_degree
        self.criterion = criterion

    def forward(self, pred, true, x):
        pred = higher_order_derivatives(pred, x, self.diff_degree)
        true = higher_order_derivatives(true, x, self.diff_degree)
        return self.criterion(pred, true)
