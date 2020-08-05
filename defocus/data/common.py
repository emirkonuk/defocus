# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/04_dataset_common.ipynb (unless otherwise specified).

__all__ = ['add_gaussian_noise', 'crop', 'hflip', 'vflip', 'rot90', 'channel_shuffle', 'saturation']

# Cell
import os
import cv2
import random
import numpy as np
from skimage.color import rgb2hsv, hsv2rgb
'''
AUGMENTATIONS
'''
def add_gaussian_noise(img, sigma_sigma=2, rgb_range=255):
    # TODO: note that this is highly unrealistic.
    # implement poisson. also a differentiable (pytorch?) version.
    # here is a discussion and scikit implementation of poisson
    # https://github.com/yu4u/noise2noise/issues/14
    '''
    image = input_
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    out = np.random.poisson(image * vals) / float(vals)
    out = np.clip(out, 0, 255).astype(np.uint8)
    '''
    # the default's are from Seungjun again
    # https://github.com/SeungjunNah/DeepDeblur-PyTorch/blob/master/src/data/common.py#L43
    sigma = np.random.normal() * sigma_sigma * rgb_range/255
    # I think casting to 32-bit here is redundant, but let's keep with Seungjun
    noise = np.random.randn(*img.shape).astype(np.float32) * sigma
    return (img + noise).clip(0, rgb_range)

def crop(img, patch_x, patch_y, crop_size=256):
    # be careful about the patch location. it should be:
    # patch_y = random.randrange(0, H-patch_size+1)
    # patch_x = random.randrange(0, W-patch_size+1)
    return img[patch_y:patch_y+crop_size, patch_x:patch_x+crop_size, :]

def hflip(img, p=0.5):
    return img[:, ::-1, :]

def vflip(img):
    return img[::-1, :, :]

def rot90(img):
    return img.transpose(1, 0, 2)

def channel_shuffle(img, rgb_order=[0,1,2]):
    return img[..., rgb_order]

def saturation(img, modifier=1.0, rgb_range=255):
    # slooooow.
    # TODO: maybe just use g(x)=αf(x)+β, i.e. contrast and brightness?
    # if not, maybe play with LAB space, it is slow anyway
    # TODO: try using cv2 instead of skimage
    # for now, I am keeping Seungjun's version
    hsv_img = rgb2hsv(img)
    hsv_img[..., 1] *= modifier
    return hsv2rgb(hsv_img).clip(0, 1) * rgb_range