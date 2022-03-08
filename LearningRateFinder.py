import torch
from ExponentialLearningRate import ExponentialLR
from IteratorWrapper import IteratorWrapper


class LRFinder:
    """
    Learning rate finder class used to determine the optimal learning rate for training data.
    """
    def __init__(self, model, optimizer, criterion, device):
        self.optimizer = optimizer
        self.model = model
        self.criterion = criterion
        self.device = device

        # Saving the inital parameters such that they do not need to be recalculated
        torch.save(model.state_dict(), 'init_params.pt')

    def range_test(self, iterator, end_lr=10, num_iter=100, smooth_f=0.05, diverge_th=5):
        lrs = []
        losses = []
        best_loss = float('inf')

        # defining an exponential learning rate and wrapping the initial
        lr_scheduler = ExponentialLR(self.optimizer, end_lr, num_iter)
        iterator = IteratorWrapper(iterator)

        for iteration in range(num_iter):
            # Calculating loss from training
            loss = self.train_batch(iterator)
            lrs.append(lr_scheduler.get_last_lr()[0])

            # updating the learning rate
            lr_scheduler.step()

            if iteration > 0:
                loss = smooth_f * loss + (1 - smooth_f) * losses[-1]

            if loss < best_loss:
                best_loss = loss

            losses.append(loss)

            if loss > diverge_th * best_loss:
                print("Stopping early, the loss has diverged")
                break

        return lrs, losses

    def train_batch(self, iterator):
        self.model.train()
        self.optimizer.zero_grad()

        x, y = iterator.get_batch()

        x = x.to(self.device)
        y = y.to(self.device)

        y_pred, _ = self.model(x)

        loss = self.criterion(y_pred, y)
        loss.backward()
        self.optimizer.step()

        return loss.item()