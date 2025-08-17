"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os


# def tensor2im(input_image, imtype=np.uint8):
#     """"Converts a Tensor array into a numpy image array.

#     Parameters:
#         input_image (tensor) --  the input image tensor array
#         imtype (type)        --  the desired type of the converted numpy array
#     """
#     if not isinstance(input_image, np.ndarray):
#         if isinstance(input_image, torch.Tensor):  # get the data from a variable
#             image_tensor = input_image.data
#         else:
#             return input_image
#         image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
#         if image_numpy.shape[0] == 1:  # grayscale to RGB
#             image_numpy = np.tile(image_numpy, (3, 1, 1))
#         image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
#     else:  # if it is a numpy array, do nothing
#         image_numpy = input_image
#     return image_numpy.astype(imtype)

# def tensor2im(input_image, imtype=np.uint8):
#     """Convert a Tensor into a numpy image array."""

#     if isinstance(input_image, torch.Tensor):
#         image_tensor = input_image.data
#     else:
#         return input_image

#     image_numpy = image_tensor[0].cpu().float().numpy()

#     # Handle images with >3 channels (e.g., 6-channel output)
#     if image_numpy.shape[0] > 3:
#         print(f"[WARN] tensor2im(): Trimming image channels from {image_numpy.shape[0]} to 3.")
#         image_numpy = image_numpy[:3]

#     # CHW to HWC
#     image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0

#     return image_numpy.astype(imtype)

def tensor2im(input_image, imtype=np.uint8):
    """Convert a Tensor into a numpy array for visualization."""
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image

    # Remove batch dimension if present
    if image_tensor.dim() == 4:
        image_tensor = image_tensor[0]

    image_numpy = image_tensor.cpu().float().numpy()

    # Handle different channel configurations
    if image_numpy.shape[0] == 1:
        # (1, H, W) → (H, W)
        image_numpy = image_numpy[0]
        image_numpy = (image_numpy + 1) / 2.0 * 255.0
        return image_numpy.astype(imtype)
    
    elif image_numpy.shape[0] == 2:
        # Just visualize first channel
        image_numpy = image_numpy[0]
        image_numpy = (image_numpy + 1) / 2.0 * 255.0
        return image_numpy.astype(imtype)

    elif image_numpy.shape[0] == 3:
        # (3, H, W) → (H, W, 3)
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        return image_numpy.astype(imtype)
    
    else:
        raise ValueError(f"Unsupported number of channels: {image_numpy.shape[0]}")



def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


# def save_image(image_numpy, image_path, aspect_ratio=1.0):
#     """Save a numpy image to the disk

#     Parameters:
#         image_numpy (numpy array) -- input numpy array
#         image_path (str)          -- the path of the image
#     """

#     image_pil = Image.fromarray(image_numpy)
#     h, w, _ = image_numpy.shape

#     if aspect_ratio > 1.0:
#         image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
#     if aspect_ratio < 1.0:
#         image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
#     image_pil.save(image_path)

def save_image(image_numpy, image_path, aspect_ratio=1.0):
    if image_numpy.ndim == 2:
        # Grayscale image
        h, w = image_numpy.shape
        image_pil = Image.fromarray(image_numpy.astype(np.uint8), mode='L')
    elif image_numpy.ndim == 3:
        h, w, _ = image_numpy.shape
        image_pil = Image.fromarray(image_numpy.astype(np.uint8))
    else:
        raise ValueError(f"Unsupported image shape: {image_numpy.shape}")

    if aspect_ratio != 1.0:
        new_w = int(w * aspect_ratio)
        image_pil = image_pil.resize((new_w, h), Image.BICUBIC)

    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
