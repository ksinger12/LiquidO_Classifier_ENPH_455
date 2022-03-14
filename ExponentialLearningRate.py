from torch.optim.lr_scheduler import _LRScheduler


class ExponentialLR(_LRScheduler):
    """
    Exponential learning rate finder class. Extends the learning rate scheduler class.
    """
    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [lr * (self.end_lr / lr) ** (self.last_epoch / self.num_iter) for lr in self.base_lrs]
