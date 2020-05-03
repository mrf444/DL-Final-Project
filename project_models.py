import os.path
import pandas as pd
import numpy as np
import math
import scipy.ndimage as ndimage
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset
from data_helper import UnlabeledDataset, LabeledDataset
from helper import collate_fn, draw_box
from skimage import measure

################################ UNet Model ######################################

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
                
class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.decode(x)
    
class _DecoderBlock2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock2, self).__init__()
        self.decode = nn.Sequential(
            #nn.Conv2d(in_channels, middle_channels, kernel_size=3),
            nn.Conv2d(in_channels, middle_channels, kernel_size=2),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            #nn.Conv2d(middle_channels, middle_channels, kernel_size=3),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=2),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.decode(x)
    
class UNet(nn.Module):
    def __init__(self,num_classes, semi_supervised=False):
        super(UNet, self).__init__()
        
        if not semi_supervised: #use regular encoder block
            self.enc1 = _EncoderBlock(3, 64)
        else: #if semi_supervised
            self.enc1 = _EncoderBlock(1, 64)
            
        self.enc2 = _EncoderBlock(64, 128)
        self.enc3 = _EncoderBlock(128, 256)
        self.enc4 = _EncoderBlock(256, 512, dropout=True)
        
        if not semi_supervised: #use regular decoder block
            self.center = _DecoderBlock(512, 1024, 512)
            self.dec4 = _DecoderBlock(1024, 512, 256)
            self.dec3 = _DecoderBlock(512, 256, 128)
            self.dec2 = _DecoderBlock(256, 128, 64)
        else: #if semi_supervised
            self.center = _DecoderBlock2(512, 1024, 512)
            self.dec4 = _DecoderBlock2(1024, 512, 256)
            self.dec3 = _DecoderBlock2(512, 256, 128)
            self.dec2 = _DecoderBlock2(256, 128, 64)
            
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(64, num_classes, kernel_size=1) #2 because binary 0/1
        initialize_weights(self)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(enc4)
        dec4 = self.dec4(torch.cat([center, F.interpolate(enc4, center.size()[2:], mode='bilinear',align_corners=True)], 1))
        dec3 = self.dec3(torch.cat([dec4, F.interpolate(enc3, dec4.size()[2:], mode='bilinear',align_corners=True)], 1))
        dec2 = self.dec2(torch.cat([dec3, F.interpolate(enc2, dec3.size()[2:], mode='bilinear',align_corners=True)], 1))
        dec1 = self.dec1(torch.cat([dec2, F.interpolate(enc1, dec2.size()[2:], mode='bilinear',align_corners=True)], 1))
        final = self.final(dec1)
        output = F.interpolate(final, 800, mode='bilinear',align_corners=True) #upsample to be 800x800
        return output

################################ End of UNet Model ###########################################
    
################################ Bounding Box Functions ######################################
def bbox_to_label(target_object):
    categories = target_object[0]['category']
    bboxes = target_object[0]['bounding_box']

    output = np.zeros((800,800))
    #print(len(categories))

    for i in range(len(bboxes)):
        class_label = 1 #categories[i]
        this_bbox = bboxes[i]
        flx, frx, blx, brx = this_bbox[0]
        fly, fry, bly, bry = this_bbox[1]
        fx = math.floor(10*((flx + frx)/2) + 400)
        bx = math.floor(10*((blx + brx)/2) + 400)
        fy = math.floor(10*((fly + bly)/2) + 400)
        by = math.floor(10*((fry + bry)/2) + 400)

        #output[fx:bx, fy:by] = class_label
        #output[bx:fx, by:fy] = class_label

        output[fy:by, fx:bx] = class_label
        output[by:fy, bx:fx] = class_label

    return output

def get_bboxes_from_output(model_output): #v2
    test_label = measure.label(model_output)
    output = test_label.copy()
    bboxes = []

    props = measure.regionprops(test_label)

    for prop in props:
        fy,fx,by,bx = prop.bbox
        fy, fx, by, bx = [min(fy,799), min(fx,799), min(by, 799), min(bx, 799)]
        flx, frx, blx, brx, fly, bly, fry, bry = (fx, fx, bx, bx, fy, fy, by, by)

        output[fy:by, fx-1:fx+1] = 50
        output[fy:by, bx-1:bx+1] = 50
        output[fy-1:fy+1, fx:bx] = 50
        output[by-1:by+1, fx:bx] = 50

        this_bbox = np.array([[flx, frx, blx, brx], [fly, fry, bly, bry]])
        this_bbox = (this_bbox - 400)/10
        bboxes.append(this_bbox)

    return torch.tensor(bboxes)

def bbox_to_label_bionary(target_object):
    categories = target_object[0]['category']
    bboxes = target_object[0]['bounding_box']

    output = np.zeros((800,800))

    for i in range(len(bboxes)):
        #class_label = categories[i]
        this_bbox = bboxes[i]
        flx, frx, blx, brx = this_bbox[0]
        fly, fry, bly, bry = this_bbox[1]
        fx = math.floor(10*((flx + frx)/2) + 400)
        bx = math.floor(10*((blx + brx)/2) + 400)
        fy = math.floor(10*((fly + bly)/2) + 400)
        by = math.floor(10*((fry + bry)/2) + 400)

        output[fy:by, fx:bx] = 1
        output[by:fy, bx:fx] = 1

    return output

def frankenstein(image_object):
    this_image = image_object[0]
    front = torch.cat((this_image[0], this_image[1], this_image[2]), 2)
    back = torch.cat((this_image[5], this_image[4], this_image[3]), 2)
    all_images = torch.cat((front, back), 1)
    all_images = all_images.unsqueeze(0)

    return all_images

################################ Pretrained Model/Feature Extractor ######################################
class Unsupervised_Model_wo_convtrans(nn.Module):
    def __init__(self):
        super(Unsupervised_Model_wo_convtrans, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size=10, stride=1)
        self.conv1_bn = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=10, kernel_size=5, stride=1)
        self.conv2_bn = nn.BatchNorm2d(10)
        self.conv3 = nn.Conv2d(in_channels=10, out_channels=3, kernel_size=5, stride=1)
        self.conv3_bn = nn.BatchNorm2d(3)
        self.linear1 = nn.Linear(in_features=110, out_features=512)
        self.linear2 = nn.Linear(in_features=177, out_features=918)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.conv1_bn(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv2(x)
        x = F.relu(self.conv2_bn(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv3(x)
        x = F.relu(self.conv3_bn(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = x.view(-1, 177, 110)  
        x = F.relu(self.linear1(x))
        x = x.view(-1, 512, 177)
        x = self.linear2(x)
      
        return x
    
################################ End of Pretrained Model/Feature Extractor ######################################

################################ Freeze/Unfreeze Methods ######################################
class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
        
def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True
        
 ################################ End of Freeze/Unfreeze Methods ######################################

