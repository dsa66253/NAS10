import torch

class ValController():
    def __init__(self, cfg, device, valDataloder, criterion):
        self.valDataloder = valDataloder
        self.cfg = cfg
        self.device = device
        self.criterion = criterion
        
    def val(self, net):

        net.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for i, data in enumerate(self.valDataloder):
                images, labels = data
                labels = labels.to(self.device)
                images = images.to(self.device)
                outputs = net(images)
                _, predict = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predict == labels).sum().item()
            acc = correct / total

        net.train()
        return acc * 100
    
    