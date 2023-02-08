import torch.nn as nn


def recon_loss(x_hat,x,m):
    criterion = nn.MSELoss()
    return criterion(x_hat*m,x*m)

def sup_loss(y_hat,y,y_mask):
    criterion = nn.MSELoss()
    return criterion(y_hat*y_mask,y*y_mask)

def matching_loss(c_hat,c):
    criterion = nn.MSELoss()
    return criterion(c_hat,c)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
