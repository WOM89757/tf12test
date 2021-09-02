import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras import layers 
from keras import models
from functools import partial

def _after_conv(in_tensor):
    norm = layers.BatchNormalization()(in_tensor)
    return layers.Activation('relu')(norm)

def conv3(in_tensor, filters):
    conv = layers.Conv2D(filters, kernel_size=3, strides=1, padding='same')(in_tensor)
    return _after_conv(conv)

def conv3_downsample(in_tensor, filters):
    conv = layers.Conv2D(filters, kernel_size=3, strides=2, padding='same')(in_tensor)
    return _after_conv(conv)

def conv1(in_tensor, filters):
    conv = layers.Conv2D(filters, kernel_size=1, strides=1, padding='same')(in_tensor)
    return _after_conv(conv)

def conv1_downsample(in_tensor, filters):
    conv = layers.Conv2D(filters, kernel_size=1, strides=2, padding='same')(in_tensor)
    return _after_conv(conv)

def bottleneck(inputs, filters, downsample):
    if downsample:
        #print('inputs %s' % inputs)
        #conv1_rb = conv1_downsample(inputs, filters[0])
        conv1_rb = conv1(inputs, filters[0])
        #print('conv1_rb %s' % conv1_rb)
    else:
        conv1_rb = conv1(inputs, filters[0])
    conv2_rb = conv3(conv1_rb, filters[1])
    conv3_rb = conv1(conv2_rb, filters[2])

    n_in = inputs.get_shape()[-1]
    if n_in != filters[2]:
        shortcut = conv1(inputs, filters[2])
        #print('n_in %s : %s shortcut %s' % (n_in, filters[2], shortcut))
    else:
        shortcut = inputs
    result = layers.Add()([conv3_rb, shortcut])
    return layers.Activation('relu')(result)


def building_block(in_tensor, filters, downsample=False):
    if downsample:
        conv1_rb = conv3_downsample(in_tensor, filters[0])
    else:
        conv1_rb = conv3(in_tensor, filters[0])
    conv2_rb = conv3(conv1_rb, filters[1])
    if downsample:
        in_tensor = conv1_downsample(in_tensor, filters[1])
    result = layers.Add()([conv2_rb, in_tensor])
    return layers.Activation('relu')(result)

def block(in_tensor, filters, n_block, netname, downsample=False):
    res = in_tensor
    for i in range(n_block):
        if i == 0:
            if netname == 'ResNet18' or netname == 'ResNet34':
                res = building_block(in_tensor, filters, downsample)
            else:
                res = bottleneck(in_tensor, filters, downsample)

        else:
            if netname == 'ResNet18' or netname == 'ResNet34':
                res = building_block(res, filters, False)
            else:
                res = bottleneck(in_tensor, filters, False)
    return res

def resnet(image_batch, netname):
    resnet_config = {
                    'ResNet18':[2, 2, 2, 2],
                    'ResNet34':[3, 4, 6, 3],
                    'ResNet50':[3, 4, 6, 3],
                    'ResNet101':[3, 4, 23, 3],
                    'ResNet152':[3, 8, 36, 3]
                    }
    layers_dims = resnet_config[netname]

    filter_block1 = [64, 64, 256]
    filter_block2 = [128, 128, 512]
    filter_block3 = [256, 256, 1024]
    filter_block4 = [512, 512, 2048]

    conv = layers.Conv2D(64, 7, strides=2, padding='same')(image_batch) 
    conv =  _after_conv(conv)
    pool1 = layers.MaxPool2D(3, 2, padding='same')(conv)
    conv1_block = block(pool1, filter_block1, layers_dims[0], netname, False)
    print(conv1_block)
    conv2_block = block(conv1_block, filter_block2, layers_dims[1], netname, True)
    print(conv2_block)
    conv3_block = block(conv2_block, filter_block3, layers_dims[2], netname, True)
    print(conv3_block)
    conv4_block = block(conv3_block, filter_block4, layers_dims[3], netname, True)
    print(conv4_block)
    pool2 = layers.GlobalAvgPool2D()(conv4_block) 
    _y = layers.Dense(1000, activation='softmax')(pool2)
    # ResNet-18 & ResNet-34 [64, 128, 256, 512]
    # [2, 2, 2, 2] & [3, 4, 6, 3]
    # ResNet-50 & ResNet-101 & ResNet-152 [256, 512, 1024, 2048]
    # [3, 4, 6, 3] & [3, 4, 23, 3] & [3, 8, 36, 3]

    return _y

x = layers.Input(shape=(224, 224, 3))
y_ = layers.Input(shape=(1000,))
#y = resnet(x, netname='ResNet34')
y = resnet(x, netname='ResNet152')
model = models.Model(x, y)
print(model.summary())
