import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time,sys
from util.tools import *

#if export onnx model, need to ONNX_EXPORT = True
ONNX_EXPORT = False
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, name, batchnorm, act='leaky'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        if act == 'leaky':
            self.act = nn.LeakyReLU()
        elif act == 'relu':
            self.act = nn.ReLU()
        
        self.module = nn.Sequential()
        self.module.add_module(name+'_conv', self.conv)
        if batchnorm == 1:
            self.module.add_module(name+"_bn", self.bn)
        if act != 'linear':
            self.module.add_module(name+"_act", self.act)

    def forward(self, x):
        return self.module(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, kernel_size = 3):
        super().__init__()
        self.conv_pointwise = nn.Conv2d(in_channels, mid_channels, kernel_size = 1)
        self.bn_pt = nn.BatchNorm2d(mid_channels)
        self.act = nn.LeakyReLU()
        self.conv = nn.Conv2d(mid_channels, in_channels, kernel_size = kernel_size, padding=1, stride=1)
        self.bn_conv = nn.BatchNorm2d(in_channels)
        self.module = nn.Sequential(self.conv_pointwise,
                                   self.bn_pt,
                                   self.act,
                                   self.conv,
                                   self.bn_conv,
                                   self.act)
    
    def forward(self, x):
        return x + self.module(x)

class ConvUp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, up_ratio):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU()        
        self.upsample = nn.Upsample(scale_factor=up_ratio, mode='nearest')
        self.module = nn.Sequential(self.conv,
                                   self.bn,
                                   self.act,
                                   self.upsample)
    def forward(self, x):
        return self.module(x)

def make_conv_layer(layer_idx, modules, layer_info, in_channel):
    filters = int(layer_info['filters'])
    size = int(layer_info['size'])
    stride = int(layer_info['stride'])
    pad = size // 2
    if layer_info['batch_normalize'] == '1':
        modules.add_module('layer_'+str(layer_idx)+'_conv',
                          nn.Conv2d(in_channel,
                                    filters,
                                    size,
                                    stride,
                                    pad,
                                    bias=False))
        modules.add_module('layer_'+str(layer_idx)+'_bn',
                          nn.BatchNorm2d(filters, momentum=0.03, eps=1E-4))
    else:
        modules.add_module('layer_'+str(layer_idx)+'_conv',
                          nn.Conv2d(in_channel,
                                    filters,
                                    size,
                                    stride,
                                    pad,
                                    bias=True))

    if layer_info['activation'] == 'leaky':
        modules.add_module('layer_'+str(layer_idx)+'_act',
                          nn.LeakyReLU(0.1, inplace=True))
    elif layer_info['activation'] == 'relu':
        modules.add_module('layer_'+str(layer_idx)+'_act',
                          nn.ReLU())
    #return modules

import torch.nn.functional as F 
class upsample_test_without_scale_factor(nn.Module):
    def forward(self, x):
        sh = torch.tensor(x.shape)
        return F.interpolate(x, size=(sh[2] * 2, sh[3] * 2), mode='nearest')

class DarkNet53(nn.Module):
    yolo_strides = [[32,32],[16,16],[8,8]]
    def __init__(self, cfg, param):
        super().__init__()
        self.batch = int(param['batch'])
        self.in_channels = int(param['in_channels'])
        self.in_width = int(param['in_width'])
        self.in_height = int(param['in_height'])
        self.n_classes = int(param['class'])
        self.module_cfg = parse_model_config(cfg)
        self.module_list = self.set_layer(self.module_cfg)
        self.yolo_layers = [layer[0] for layer in self.module_list if isinstance(layer[0], self.YoloLayer)]
        self.box_per_anchor = 3
        self.yolo_grid_size = [[self.in_width // 32, self.in_height // 32], [self.in_width // 16, self.in_height // 16],[self.in_width // 8, self.in_height // 8]]
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)
    class YoloLayer(nn.Module):
        def __init__(self, yolo_idx, layer_info, stride, in_width, in_height):
            super().__init__()
            self.n_classes = int(layer_info['classes'])
            self.ignore_thresh = float(layer_info['ignore_thresh'])
            self.box_attr = self.n_classes + 5
            mask_idxes = [int(x) for x in layer_info["mask"].split(",")]
            anchor_all = [int(x) for x in layer_info["anchors"].split(",")]
            anchor_all = [(anchor_all[i],anchor_all[i+1]) for i in range(0,len(anchor_all),2)]
            self.anchor = torch.tensor([anchor_all[x] for x in mask_idxes])
            self.in_width = in_width
            self.in_height = in_height
            self.stride = torch.tensor(stride, dtype=torch.int8)
            self.lw = None
            self.lh = None

        def forward(self, x):
            if ONNX_EXPORT:
                return x
            self.lw, self.lh = x.shape[3], x.shape[2]
            self.anchor = self.anchor.to(x.device)
            self.stride = self.stride.to(x.device)
            # reshape input tensor from 4-dim to 5-dim. 
            # from [batch, num_anchor * num_attributes, x_height, x_width]
            # to   [batch, num_anchor, x_height, x_width, num_attributes]
            x = x.view(-1,self.anchor.shape[0],self.box_attr,self.lh,self.lw).permute(0,1,3,4,2).contiguous()
            if not self.training:
                anchor_grid = self.anchor.view(1,-1,1,1,2).to(x.device)
                grids = self._make_grid(self.lw, self.lh).to(x.device)
                #Get outputs
                x[...,0:2] = (torch.sigmoid(x[...,0:2]) + grids) * self.stride #center xy
                x[...,2:4] = torch.exp(x[...,2:4]) * anchor_grid     # Width Height
                x[...,4:] = torch.sigmoid(x[...,4:])       # Conf, Class
                x = x.view(x.shape[0], -1, self.box_attr)
            return x

        def _make_grid(self, nx=20, ny=20):
            yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)]) #, indexing='ij'
            return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def set_layer(self, layer_info):
        module_list = nn.ModuleList()
        in_channels = [self.in_channels]
        yolo_idx = 0
        for layer_idx, info in enumerate(layer_info):
            modules = nn.Sequential()
            if info['type'] == "convolutional":
                make_conv_layer(layer_idx, modules, info, in_channels[-1])
                in_channels.append(int(info['filters']))
            elif info['type'] == 'shortcut':
                modules.add_module('layer_'+str(layer_idx)+'_shortcut', nn.Identity())
                in_channels.append(in_channels[-1])
            elif info['type'] == 'route':
                modules.add_module('layer_'+str(layer_idx)+'_route', nn.Identity())
                layers = [int(y) for y in info["layers"].split(",")]
                if len(layers) == 1:
                    in_channels.append(in_channels[layers[0]])
                elif len(layers) == 2:
                    in_channels.append(in_channels[layers[0]] + in_channels[layers[1]+1])
            elif info['type'] == 'upsample':
                if ONNX_EXPORT:
                    modules.add_module('layer_'+str(layer_idx)+'_upsample',
                                       upsample_test_without_scale_factor())
                else:
                    modules.add_module('layer_'+str(layer_idx)+'_upsample',
                                       nn.Upsample(scale_factor=int(info['stride']), mode='nearest'))
                in_channels.append(in_channels[-1])
            elif info['type'] == 'yolo':
                yololayer = self.YoloLayer(yolo_idx, info, self.yolo_strides[yolo_idx], self.in_width, self.in_height)
                modules.add_module('layer_'+ str(layer_idx)+'_yolo', yololayer)
                in_channels.append(in_channels[-1])
                yolo_idx += 1
            elif info['type'] == 'maxpool':
                _pad = int(int(info['size']) - 2)
                if int(info['stride']) == 1:
                    submodule = nn.Sequential(nn.ConstantPad2d((0,1,0,1),-50),
                                              nn.MaxPool2d(kernel_size=int(info['size']), stride=int(info['stride']), padding=_pad))
                    modules.add_module('layer_'+str(layer_idx)+'_maxpool',
                                        submodule)
                else:
                    modules.add_module('layer_'+str(layer_idx)+'_maxpool',
                                        nn.MaxPool2d(kernel_size=int(info['size']), stride=int(info['stride']), padding=_pad))
                in_channels.append(in_channels[-1])
            module_list.append(modules)
        return module_list
    
    def initialize_weights(self):
        # track all layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def get_grid_wh(self, grid_idx):
        grid_w, grid_h = self.yolo_grid_size[grid_idx]
        w_per_grid, h_per_grid = self.in_width // grid_w, self.in_height // grid_h
        return [w_per_grid, h_per_grid]
    
    def forward(self, x):
        layer_result = []
        yolo_result = []
        for idx, (name, layer) in enumerate(zip(self.module_cfg, self.module_list)):
            if name['type'] == 'convolutional':
                x = layer(x)
                layer_result.append(x)
            elif name['type'] == 'shortcut':
                x = x + layer_result[int(name['from'])]
                layer_result.append(x)
            elif name['type'] == 'yolo':
                yolo_x = layer(x)
                layer_result.append(yolo_x)
                yolo_result.append(yolo_x)
            elif name['type'] == 'upsample':
                x = layer(x)
                layer_result.append(x)
            elif name['type'] == 'route':
                layers = [int(y) for y in name["layers"].split(",")]
                features = [layer_result[l] for l in layers]
                x = torch.cat(features, 1)
                layer_result.append(x)
            elif name['type'] == 'maxpool':
                x = layer(x)
                layer_result.append(x)
        if ONNX_EXPORT:
            return yolo_result
        else:
            return yolo_result if self.training else torch.cat(yolo_result, dim=1)

    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""
        ### weight가 binary로 되어 있어서 추가함
        
        # Open the weights file
        with open(weights_path, "rb") as f:
            # First five are header values
            self.version = np.fromfile(f, dtype=np.int32, count=3)  # (int32) version info: major, minor, revision
            self.seen = np.fromfile(f, dtype=np.int64, count=1)  # (int64) number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights
            print(self.version, self.seen)
        # Establish cutoff for loading backbone weights
        cutoff = None
        # If the weights file has a cutoff, we can find out about it by looking at the filename
        # examples: darknet53.conv.74 -> cutoff is 74
        filename = os.path.basename(weights_path)
        if ".conv." in filename:
            try:
                cutoff = int(filename.split(".")[-1])  # use last part of filename
            except ValueError:
                pass

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_cfg, self.module_list)):
            # print(i, module_def)
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv.bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(
                    weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w
                # print("total : {}, now {}".format(len(weights), ptr))
    
    def save_darknet_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        self.version = np.array([0, 2, 5], dtype=np.int32)  # (int32) version info: major, minor, revision
        self.seen = np.array([0], dtype=np.int64)  # (int64) number of images seen during training
        fp = open(path, "wb")
        self.version.tofile(fp)
        self.seen.tofile(fp)
        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_cfg[:cutoff], self.module_list[:cutoff])):
            #print(i, module_def, module)
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                    num_b = conv_layer.bias.numel()
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)
                num_w = conv_layer.weight.numel()
        fp.close()