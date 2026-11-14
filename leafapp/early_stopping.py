# Decides when to stop training early, once the validation loss stops improving.

class EarlyStopping:
    """
    Tracks the validation loss and says when training should stop.

    Args:
        patience: how many epochs in a row the validation loss may fail
        to beat its best value before training stops.
    """
    def __init__(self, patience):
        self.patience = patience
        self.best_loss = float("inf")
        self.epochs_without_improvement = 0
        
    def step(self, val_loss):
        """
        Record this epoch's validation loss.

        A loss lower than the best so far resets the counter; any other loss adds 1 to it.
        Returns True once the counter reaches patience (training should stop), otherwise False.
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        if self.epochs_without_improvement >= self.patience:
            return True
        else:
            return False
            