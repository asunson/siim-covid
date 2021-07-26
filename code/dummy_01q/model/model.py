from common import *
from siim import *

from timm.models.efficientnet import *

class Net(nn.Module):
    def __init__(self, model_layers):
        super(Net, self).__init__()
        self.model_layers = model_layers

    # @torch.cuda.amp.autocast()
    def forward(self, image):
        batch_size = len(image)
        x = 2*image-1     # ; print('input ',   x.shape)

        # assumes final line in config is the logit layer
        for layerName in self.model_layers.keys():
            layer = self.model_layers[layerName]
            if ("mask" in layerName):
                mask = layer.func(x)
            elif ("logit" in layerName):
                logit = layer.func(x)
            elif ("reshape" in layerName): 
                # needs to be reshaped to batch size which only the class knows about so very custom logic here
                x = layer.func(x, 1).reshape(batch_size,-1)
            else:
                x = layer.func(x)

        return mask, logit

# check #################################################################

def run_check_net():
    batch_size = 2
    C, H, W = 3, 512, 512
    #C, H, W = 3, 640, 640
    image = torch.randn(batch_size, C, H, W).cuda()
    mask  = torch.randn(batch_size, num_study_label, H, W).cuda()

    net = Net().cuda()
    logit, mask = net(image)

    print(image.shape)
    print(logit.shape)
    print(mask.shape)


# main #################################################################
if __name__ == '__main__':
    run_check_net()


