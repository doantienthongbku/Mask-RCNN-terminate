import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor

# import function and class support from other file
from loss.loss import *


########################################################
# Support function and support class
########################################################
class SamePad2d(nn.Module):
    """Mimics tensorflow's 'SAME' padding.
    """

    def __init__(self, kernel_size, stride):
        super(SamePad2d, self).__init__()
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        self.stride = torch.nn.modules.utils._pair(stride)

    def forward(self, input):
        in_width = input.size()[2]
        in_height = input.size()[3]
        out_width = math.ceil(float(in_width) / float(self.stride[0]))
        out_height = math.ceil(float(in_height) / float(self.stride[1]))
        pad_along_width = ((out_width - 1) * self.stride[0] +
                           self.kernel_size[0] - in_width)
        pad_along_height = ((out_height - 1) * self.stride[1] +
                            self.kernel_size[1] - in_height)
        pad_left = math.floor(pad_along_width / 2)
        pad_top = math.floor(pad_along_height / 2)
        pad_right = pad_along_width - pad_left
        pad_bottom = pad_along_height - pad_top
        return F.pad(input, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)

    def __repr__(self):
        return self.__class__.__name__


########################################################
# Backbone
########################################################

class ResNet50_extract_feature(nn.Module):
    """Extract feature from ResNet50
    """
    def __init__(self):
        super(ResNet50_extract_feature, self).__init__()

        net = models.ResNet50(pretrained=True)
        # create feature extractor from ResNet50 network
        self.body = create_feature_extractor(
            net, return_nodes={f'layer{k}': str(v)
                             for v, k in enumerate([1, 2, 3, 4])})
        
    def forward(self, x):
        x = self.body(x)
        return x

    def stages(self):
        return self.body


########################################################
# Region Proposal Network
########################################################

class RPN(nn.Module):
    """RPN Network

    Args:
        anchors_per_location (int): numbers of anchor per location
        anchor_stride (int): stride of anchors
        depth (int): the depth of tensor before pass to RPN
    """
    
    def __init__(self, anchors_per_location, anchor_stride, depth):

        super(RPN, self).__init__()
        self.anchors_per_location = anchors_per_location
        self.anchor_stride = anchor_stride
        self.depth = depth

        self.padding = SamePad2d(kernel_size=3, stride=self.anchor_stride)
        self.conv_shared = nn.Conv2d(self.depth, 512, kernel_size=3, stride=self.anchor_stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv_class = nn.Conv2d(512, 2 * anchors_per_location, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=2)
        self.conv_bbox = nn.Conv2d(512, 4 * anchors_per_location, kernel_size=1, stride=1)

    def forward(self, x):
        # Shared convolutional base of the RPN
        x = self.relu(self.conv_shared(self.padding(x)))

        # Anchor Score. [batch, anchors per location * 2, height, width].
        rpn_class_logits = self.conv_class(x)

        # Reshape to [batch, 2, num_anchors]
        rpn_class_logits = rpn_class_logits.permute(0,2,3,1)            # [batch, height, width, anchors per location * 2]
        rpn_class_logits = rpn_class_logits.contiguous()                # contiguous array is just an array stored in an unbroken block of memory
        rpn_class_logits = rpn_class_logits.view(x.size()[0], -1, 2)    # [batch, 2, num_anchors]

        # Softmax on last dimension of BG/FG.
        rpn_probs = self.softmax(rpn_class_logits)

        # Bounding box refinement. [batch, H, W, anchors per location, depth]
        # where depth is [x, y, log(w), log(h)]
        rpn_bbox = self.conv_bbox(x)

        # Reshape to [batch, 4, num_anchors]
        rpn_bbox = rpn_bbox.permute(0,2,3,1)
        rpn_bbox = rpn_bbox.contiguous()
        rpn_bbox = rpn_bbox.view(x.size()[0], -1, 4)        # [batch, 4, num_anchors]

        return [rpn_class_logits, rpn_probs, rpn_bbox]


