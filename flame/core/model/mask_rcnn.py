import math

import numpy as np
from roi_align import RoIAlign
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import os

# import function and class support from other file
from loss.loss import *
import utils
from backbone.resnet50 import *
from roi_align import *
from core.config import Config
from RPN import *
from head import *

config = Config()


class MaskRCNN(nn.Module):

    def __init__(self, config, model_dir):
        super(MaskRCNN, self).__init__()
        self.config = config
        self.model_dir = model_dir


    def build(self, config):
        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")
        
        # build the shared convolutional layers
        self.features_map = ResNet50_extract_feature()   # get backbone is resnet50 - C4
        self.anchors = Variable(torch.from_numpy(utils.generate_anchors(config.RPN_ANCHOR_SCALES,
                                                                        config.RPN_ANCHOR_RATIOS,
                                                                        config.BACKBONE_SHAPES,
                                                                        config.BACKBONE_STRIDES,
                                                                        config.RPN_ANCHOR_STRIDE)).float(), requires_grad=False)
        
        # Set GPU for model
        if self.config.GPU_COUNT:
            self.anchors = self.anchors.cuda()
        
                # RPN
        self.rpn = RPN(len(config.RPN_ANCHOR_RATIOS), config.RPN_ANCHOR_STRIDE, 256)

        # FPN Classifier
        self.classifier = Classifier(256, config.POOL_SIZE, config.IMAGE_SHAPE, config.NUM_CLASSES)

        # FPN Mask
        self.mask = Mask(256, config.MASK_POOL_SIZE, config.IMAGE_SHAPE, config.NUM_CLASSES)
        
        # Fix batch norm layers
        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad = False

        self.apply(set_bn_fix)

    def initialize_weights(self):
        """Initialize model weights.
        """

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                
                
    def load_weights(self, filepath):
        """Modified version of the correspoding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exlude: list of layer names to excluce
        """
        if os.path.exists(filepath):
            state_dict = torch.load(filepath)
            self.load_state_dict(state_dict, strict=False)
        else:
            print("Weight file not found ...")

        # Update the log directory
        self.set_log_dir(filepath)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
                
    
    def forward(self, x):
        self.build(self.config)
        out = self.features_map(x)
        return out
