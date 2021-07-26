from common import *
from siim import *
from timm.models.efficientnet import *
from .model import run_train

#--- example usage ---#

# base model import
e = efficientnet_b3a(pretrained=True, drop_rate=0.3, drop_path_rate=0.2)

# layer definitions
    # x = self.b0(x) #; print (x.shape)  # torch.Size([2, 40, 256, 256])
    # x = self.b1(x) #; print (x.shape)  # torch.Size([2, 24, 256, 256])
    # x = self.b2(x) #; print (x.shape)  # torch.Size([2, 32, 128, 128])
    # x = self.b3(x) #; print (x.shape)  # torch.Size([2, 48, 64, 64])
    # x = self.b4(x) #; print (x.shape)  # torch.Size([2, 96, 32, 32])
    # x = self.b5(x) #; print (x.shape)  # torch.Size([2, 136, 32, 32])
    # #------------
    # mask = self.mask(x)
    # #-------------
    # x = self.b6(x) #; print (x.shape)  # torch.Size([2, 232, 16, 16])
    # x = self.b7(x) #; print (x.shape)  # torch.Size([2, 384, 16, 16])
    # x = self.b8(x) #; print (x.shape)  # torch.Size([2, 1536, 16, 16])
    # x = F.adaptive_avg_pool2d(x,1).reshape(batch_size,-1)
    # #x = F.dropout(x, 0.5, training=self.training)
    # logit = self.logit(x)
    # return logit, mask

# assumes order of dict is desiredlayers
model_layers = {
    "layer1": {
        "func": nn.Sequential(
            e.conv_stem,
            e.bn1,
            e.act1,
        )
    },
    "layer2": {
        "func": e.blocks[0]
    },
    "layer3": {
        "func": e.blocks[1]
    },
    "layer4": {
        "func": e.blocks[2]
    },
    "layer5": {
        "func": e.blocks[3]
    },
    "layer6": {
        "func": e.blocks[4]
    },
    # the mask layer can be placed anywheree - note the shape of the preceding layer however
    # name must contain "mask"
    "mask": {
        "func": nn.Sequential(
            nn.Conv2d(136, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, padding=0),
        )
    },
    "layer7": {
        "func": e.blocks[5]
    },
    "layer8": {
        "func": e.blocks[6]
    },
    "layer9": {
        "func": nn.Sequential(
            e.conv_head, #384, 1536
            e.bn2,
            e.act2,
        )
    },
    # arbitrary layer to handle the existing adaptive_avg_pool2d function call
    # needs to be reshaped to batch size which only the class knows about so very custom logic here
    # needs to contain the word "reshape"
    "layer10-reshape": {
        "func": F.adaptive_avg_pool2d
    },
    # assumes final line in config is the logit layer
    # name must contain "logit"
    "logit": {
        "func": nn.Linear(1536,num_study_label)
    }
}

run_train(model_layers)
