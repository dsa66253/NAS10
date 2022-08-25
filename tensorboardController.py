from torch.utils.tensorboard import SummaryWriter
class AlphasMonitor():
    def __init__(self):
        self.allAlphas = 0
        self.allAlphasGrad = 0

