import torch
import torchvision.transforms as T
class TransformImgTester():
    def __init__(self, batch_size, kth):
        self.batch_size = batch_size
        self.rotater = T.RandomRotation(degrees=30)
        self.kth = kth
    def compare(self, net, train_images, predicts, train_labels, writer, iteration):
        train_transform_images = self.rotater(train_images)
        train_transform_outputs = net(train_transform_images)

        _, predictsTransform = torch.max(train_transform_outputs.data, 1)
        predictsTransformTF = (train_labels == predictsTransform)
        predictsTF = (train_labels == predicts)
        # print("iteration", iteration)
        # print("predictsTransformTF", predictsTransformTF, sep="\n")
        # print("predictsTF", predictsTF, sep="\n")
        writer.add_images('my_image_batch', train_images[:, :, :, :], 0)
        numOfWrongPredict = 0
        # print("numOfWrongPredict", numOfWrongPredict)
        first = True
        tmp = None
        for i in range(len(predictsTransformTF)):
            if (predictsTF[i].data==True and predictsTransformTF[i].data==False):
                numOfWrongPredict = numOfWrongPredict + 1
                if first==True:
                    tmp = torch.cat((train_images[i], train_transform_images[i]), dim=0)
                    first=False
                else:
                    tmp = torch.cat((tmp, train_images[i], train_transform_images[i]), dim=0)
        # tmp = torch.rand((2*numOfWrongPredict, 3, 128, 128))
        if tmp!=None:
            tmp = torch.reshape(tmp, (2*numOfWrongPredict, 3, 128, 128))
        # print("reshpae tmp.shape", tmp.shape)
        # print("tmp[1][1]", tmp[1][1])
        
        # writer.add_images("helo", tmp, 0)
            writer.add_images('{}th_iter{}'.format(self.kth, iteration), tmp, 0)
        # tmp1 =  train_images[coincident_output, :, :, :]
        # print("tmp1.shape", tmp1.shape)
        
        # writer.add_images('my_image_batch1', tmp1, 0)