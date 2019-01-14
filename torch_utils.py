import torch
import numpy as np

import math
import numbers
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import scipy.stats as st
from np_utils import *

def gkern(kernlen=5, nsig=5):
    # Code from https://stackoverflow.com/a/29731818
    """Returns a 2D Gaussian kernel array."""
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    print (x, "<<<<<<<<<<<<DDDDDDDDDDD")
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()



        kernel_size = sigma*10.#int(sigma*10.)

        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )








        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1.) / 2.
            mgrid = mgrid - mean
            mgrid = mgrid/1.5

            # print size, std, mgrid, "<<<<<<<<<<"
            kernel *= 1. / (std * math.sqrt(2. * math.pi)) * \
                      torch.exp(-(((mgrid - 0.) / (std)) ** 2)*0.5)
            # print kernel
            # print kernel.size(), "sss"


        # vec = kernel[0:200, 201].data.cpu().numpy()
        # vec = (vec<np.max(vec)/100.).astype(np.float32)
        # kernel_sz = int(201 - np.sum(vec))
        # kernel = kernel[200-kernel_sz:201+kernel_sz, 200-kernel_sz:201+kernel_sz]


        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)


        print ("Size of Gauassian filter:", kernel.size())
        # k = kernel.data.cpu().numpy()
        # k = k/k.max()
        # plt.imshow(k)
        # plt.show()
        # exit()
        # kernel = torch.from_numpy(kernel).type(torch.float32)


        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, * kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))




        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )


    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """

        padd0 = self.weight.size()[2]//2

        evenorodd = 1 - self.weight.size()[2] % 2


        input1 = F.pad(input, (padd0-evenorodd, padd0, padd0-evenorodd, padd0), "constant", 0)
        input1 = self.conv(input1, weight=self.weight, groups=self.groups)
        return input1



def rgb2lab(rgb):

    T = 0.008856

    R = rgb[0, :, :].view(1, -1)
    G = rgb[1, :, :].view(1, -1)
    B = rgb[2, :, :].view(1, -1)


    RGB = torch.cat((R, G, B), 0).cuda()

    MAT = torch.from_numpy(
        np.array([[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169], [0.019334, 0.119193, 0.950227]],
                 np.float32)).cuda()

    XYZ = torch.matmul(MAT, RGB)

    X = XYZ[0, :] / 0.950456
    Y = XYZ[1, :]
    Z = XYZ[2, :] / 1.088754

    XT = (X > T).float()
    YT = (Y > T).float()
    ZT = (Z > T).float()

    Y3 = Y ** (1. / 3.)

    fX = XT * (X ** (1. / 3.)) + (1. - XT) * (7.787 * X + 16. / 116.)
    fY = YT * Y3 + (1. - YT) * (7.787 * Y + 16. / 116.)
    fZ = ZT * (Z ** (1. / 3.)) + (1. - ZT) * (7.787 * Z + 16. / 116.)

    L = YT * (116. * Y3 - 16.0) + (1. - YT) * (903.3 * Y)
    a = 500. * (fX - fY)
    b = 200. * (fY - fZ)



    L = torch.clamp(L, 0., 100.)
    a = torch.clamp(a, -127., 127.)
    b = torch.clamp(b, -127., 127.)

    L = L/100.
    a = a/(127.*2.)
    b = b/(127.*2.)



    lab = torch.cat((L.view(1, 224, 224), a.view(1, 224, 224), b.view(1, 224, 224)), 0)
    # lab = torch.cat((L.view(1, -1), a.view(1, -1), b.view(1, -1)), 0)

    return lab


def lab2rgb(lab):
    # Thresholds
    T1 = 0.008856
    T2 = 0.206893

    L = lab[0, :].view(1, -1)*100.
    a = lab[1, :].view(1, -1)*(127.*2.)
    b = lab[2, :].view(1, -1)*(127.*2.)

    L = torch.clamp(L, min=0., max=100.)

    # Compute Y
    fY = ((L + 16.) / 116.) ** 3.
    YT = (fY > T1).float()
    fY = (1. - YT) * (L / 903.3) + (YT * fY)
    Y = fY

    # Alter fY slightly for further calculations
    fY = YT * (fY ** (1. / 3.)) + (1. - YT) * (7.787 * fY + 16. / 116.)
    # Compute X
    fX = a / 500. + fY
    XT = (fX > T2).float()
    X = XT * (fX ** 3.) + (1. - XT) * ((fX - 16. / 116.) / 7.787)

    # Compute Z
    fZ = fY - b / 200.
    ZT = (fZ > T2).float()
    Z = ZT * (fZ ** 3.) + (1. - ZT) * ((fZ - 16. / 116.) / 7.787)

    # Normalize for D65 white point
    X = X * 0.950456
    Z = Z * 1.088754

    XYZ = torch.cat((X.view(1, -1), Y.view(1, -1), Z.view(1, -1)), 0)

    MAT = torch.from_numpy(
        np.array([[3.240479, -1.537150, -0.498535], [-0.969256, 1.875992, 0.041556], [0.055648, -0.204043, 1.057311]],
                 np.float32)).cuda()

    RGB = torch.matmul(MAT, XYZ)

    RGB = torch.min(RGB, torch.ones_like(RGB))
    RGB = torch.max(RGB, torch.zeros_like(RGB))

    R = RGB[0, :].view(1, 224, 224)
    G = RGB[1, :].view(1, 224, 224)
    B = RGB[2, :].view(1, 224, 224)

    out = torch.cat((R, G, B), 0)
    out = torch.clamp(out, 0., 1.)
    return out

def torch2img(x):
    x1 = x.data.cpu().numpy()
    x1 = np.transpose(x1, [1, 2, 0])
    x1 = x1.clip(min=0.0, max=1.)
    return x1


import numpy as np
import torch
import copy


def clip_image_values(x, minv, maxv):

    x = torch.max(x, minv)
    x = torch.min(x, maxv)
    return x


def valid_bounds(img, delta=255):

    im = copy.deepcopy(np.asarray(img))
    im = im.astype(np.int)

    # General valid bounds [0, 255]
    valid_lb = np.zeros_like(im)
    valid_ub = np.full_like(im, 255)

    # Compute the bounds
    lb = im - delta
    ub = im + delta

    # Validate that the bounds are in [0, 255]
    lb = np.maximum(valid_lb, np.minimum(lb, im))
    ub = np.minimum(valid_ub, np.maximum(ub, im))

    # Change types to uint8
    lb = lb.astype(np.uint8)
    ub = ub.astype(np.uint8)

    return lb, ub


def inv_tf(x, mean, std):

    for i in range(len(mean)):

        x[i] = np.multiply(x[i], std[i], dtype=np.float32)
        x[i] = np.add(x[i], mean[i], dtype=np.float32)

    x = np.swapaxes(x, 0, 2)
    x = np.swapaxes(x, 0, 1)

    return x


def inv_tf_pert(r):

    pert = np.sum(np.absolute(r), axis=0)
    pert[pert != 0] = 1

    return pert


def get_label(x):
    s = x.split(' ')
    label = ''
    for l in range(1, len(s)):
        label += s[l] + ' '

    return label


def nnz_pixels(arr):
    return np.count_nonzero(np.sum(np.absolute(arr), axis=0))

def smooth_clip(x, v, smoothing, max_iters=1000):


    n = 1.

    epsilon = 0.#1e-2

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    t_max_r = (1. - mean[0]) / std[0] - epsilon  # 2.248
    t_max_g = (1. - mean[1]) / std[1] - epsilon # 2.428
    t_max_b = (1. - mean[2]) / std[2] - epsilon # 2.640
    t_min_r = (- mean[0]) / std[0] + epsilon# 2.248
    t_min_g = (- mean[1]) / std[1] + epsilon# 2.428
    t_min_b = (- mean[2]) / std[2] + epsilon# 2.640


    test_x = copy.deepcopy(x)
    v_i = copy.deepcopy(v)
    iter_i = 0
    while n > 0 and iter_i<max_iters:

        result_img = test_x + v_i
        result_img_r = result_img[0, 0, :, :]
        result_img_g = result_img[0, 1, :, :]
        result_img_b = result_img[0, 2, :, :]

        overshoot_r = ((result_img_r - t_max_r) >= 0).type(torch.float32).view([1, 1, 224, 224])
        overshoot_g = ((result_img_g - t_max_g) >= 0).type(torch.float32).view([1, 1, 224, 224])
        overshoot_b = ((result_img_b - t_max_b) >= 0).type(torch.float32).view([1, 1, 224, 224])


        belowshoot_r = ((result_img_r - t_min_r) <= 0).type(torch.float32).view([1, 1, 224, 224])
        belowshoot_g = ((result_img_g - t_min_g) <= 0).type(torch.float32).view([1, 1, 224, 224])
        belowshoot_b = ((result_img_b - t_min_b) <= 0).type(torch.float32).view([1, 1, 224, 224])



        alpha = .1

        ov_r_max = (result_img_r - t_max_r).data.cpu().numpy()*alpha
        ov_g_max = (result_img_g - t_max_g).data.cpu().numpy()*alpha
        ov_b_max = (result_img_b - t_max_b).data.cpu().numpy()*alpha

        bl_r_max = (result_img_r - t_min_r).data.cpu().numpy()*alpha * -1.
        bl_g_max = (result_img_g - t_min_g).data.cpu().numpy()*alpha * -1.
        bl_b_max = (result_img_b - t_min_b).data.cpu().numpy()*alpha * -1.

        ov_r_max = np.maximum(ov_r_max.max(), 0.1)
        ov_g_max = np.maximum(ov_g_max.max(), 0.1)
        ov_b_max = np.maximum(ov_b_max.max(), 0.1)

        bl_r_max = np.maximum(bl_r_max.max(), 0.1)
        bl_g_max = np.maximum(bl_g_max.max(), 0.1)
        bl_b_max = np.maximum(bl_b_max.max(), 0.1)

        # ov_r_max = np.minimum(ov_r_max, 0.05)
        # ov_g_max = np.minimum(ov_g_max, 0.05)
        # ov_b_max = np.minimum(ov_b_max, 0.05)

        # print ov_r_max, ov_g_max, ov_b_max
        # print bl_r_max, bl_g_max, bl_b_max


        overshoot_r = overshoot_r.repeat(1, 3, 1, 1)
        overshoot_g = overshoot_g.repeat(1, 3, 1, 1)
        overshoot_b = overshoot_b.repeat(1, 3, 1, 1)

        belowshoot_r = belowshoot_r.repeat(1, 3, 1, 1)
        belowshoot_g = belowshoot_g.repeat(1, 3, 1, 1)
        belowshoot_b = belowshoot_b.repeat(1, 3, 1, 1)


        overshoot_r = smoothing(overshoot_r)
        overshoot_g = smoothing(overshoot_g)
        overshoot_b = smoothing(overshoot_b)


        belowshoot_r = smoothing(belowshoot_r)
        belowshoot_g = smoothing(belowshoot_g)
        belowshoot_b = smoothing(belowshoot_b)
        # overshoot_r = overshoot_r.data.cpu().numpy()
        # overshoot_g = overshoot_g.data.cpu().numpy()
        # overshoot_b = overshoot_b.data.cpu().numpy()
        #
        #
        # print overshoot_r.shape
        # plt.subplot(131)
        # plt.imshow(overshoot_r[0, 0, :, :].data.cpu().numpy())
        # plt.subplot(132)
        # plt.imshow(normalize(usmooth_r[0, 0, :, :].data.cpu().numpy()))
        # plt.subplot(133)
        # plt.imshow(overshoot_b[0, 0, :, :].data.cpu().numpy())
        # plt.show()
        # exit()


        maxx_ov = torch.max(overshoot_r.max(), torch.max(overshoot_b.max(), overshoot_g.max())) + 1e-12
        maxx_bl = torch.max(belowshoot_r.max(), torch.max(belowshoot_g.max(), belowshoot_b.max())) + 1e-12

        # print maxx, overshoot_r.max(), overshoot_g.max(), overshoot_b.max()

        overshoot_r = overshoot_r / maxx_ov  # (overshoot_r>0.).type(torch.float32)
        overshoot_g = overshoot_g / maxx_ov  # (overshoot_r>0.).type(torch.float32)
        overshoot_b = overshoot_b / maxx_ov  # (overshoot_r>0.).type(torch.float32)

        belowshoot_r = belowshoot_r / maxx_bl
        belowshoot_g = belowshoot_g / maxx_bl
        belowshoot_b = belowshoot_b / maxx_bl

        # plt.subplot(121)
        # plt.imshow((overshoot_r[0, 0, :, :].data.cpu().numpy()))


        # overshoot_r = (overshoot_r>0.02).type(torch.float32)
        # overshoot_g = (overshoot_g>0.02).type(torch.float32)
        # overshoot_b = (overshoot_b>0.02).type(torch.float32)


        # plt.subplot(122)
        # plt.imshow((overshoot_r[0, 0, :, :].data.cpu().numpy()))
        # plt.show()



        # rr = copy.deepcopy(r_i)
        v_i[0, 0, :, :] = v_i[0, 0, :, :] - overshoot_r[0, 0, :, :] * ov_r_max + belowshoot_r[0, 0, :, :] * bl_r_max
        v_i[0, 1, :, :] = v_i[0, 1, :, :] - overshoot_g[0, 0, :, :] * ov_g_max + belowshoot_g[0, 0, :, :] * bl_g_max
        v_i[0, 2, :, :] = v_i[0, 2, :, :] - overshoot_b[0, 0, :, :] * ov_b_max + belowshoot_b[0, 0, :, :] * bl_b_max
        result_img = test_x + v_i
        result_img_r = result_img[0, 0, :, :]
        result_img_g = result_img[0, 1, :, :]
        result_img_b = result_img[0, 2, :, :]

        # print t_max_r, t_max_g, t_max_b

        overshoot_r = ((result_img_r - t_max_r) >= 0).type(torch.float32)
        overshoot_g = ((result_img_g - t_max_g) >= 0).type(torch.float32)
        overshoot_b = ((result_img_b - t_max_b) >= 0).type(torch.float32)

        belowshoot_r = ((result_img_r - t_min_r) <= 0).type(torch.float32)
        belowshoot_g = ((result_img_g - t_min_g) <= 0).type(torch.float32)
        belowshoot_b = ((result_img_b - t_min_b) <= 0).type(torch.float32)

        n_ov_r = overshoot_r.sum().item()
        n_ov_g = overshoot_g.sum().item()
        n_ov_b = overshoot_b.sum().item()

        n_bl_r = belowshoot_r.sum().item()
        n_bl_g = belowshoot_g.sum().item()
        n_bl_b = belowshoot_b.sum().item()

        # print "n_ov_r:", n_ov_r, "n_ov_g:", n_ov_g, "n_ov_b", n_ov_b, "n_ov_total:", n_ov_r+n_ov_g+n_ov_b
        # print "n_bl_r:", n_bl_r, "n_bl_g:", n_bl_g, "n_bl_b", n_bl_b, "n_bl_total:", n_bl_r+n_bl_g+n_bl_b


        n = n_ov_r + n_ov_g + n_ov_b + n_bl_r + n_bl_g + n_bl_b
        iter_i += 1

    print (iter_i, "iter")


    # result_img = test_x + v_i


    # print "-----------------"
    # print t_min_r, t_max_r
    # print torch.min(result_img[0, 0, :, :]).item(), torch.max(result_img[0, 0, :, :]).item()
    # print "-----------------"
    # print t_min_g, t_max_g
    # print torch.min(result_img[0, 1, :, :]).item(), torch.max(result_img[0, 1, :, :]).item()
    # print "-----------------"
    # print t_min_b, t_max_b
    # print torch.min(result_img[0, 2, :, :]).item(), torch.max(result_img[0, 2, :, :]).item()




    return v_i

def smooth_clip_v2(x, v, smoothing, max_iters=4000):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    epsilon = 1e-2
    test_x = copy.deepcopy(x)
    v_i = copy.deepcopy(v)
    iter_i = 0

    # deprocess x to be in [0, 1]
    test_x = deprocess_channels(test_x, mean=mean, std=std)

    # deprocess perturbation
    v_i = deprocess_channels(v_i, mean=[0., 0., 0.], std=std)

    n = 1.

    while n > 0 and iter_i<max_iters:

        result_img = test_x + v_i

        overshoot = ((result_img - 1.) >= 0).type(torch.float32)
        belowshoot = ((result_img - 0.) <= 0).type(torch.float32)



        ov_max = (result_img - 1.).data.cpu().numpy()*0.1
        bl_max = (result_img - 0.).data.cpu().numpy()*0.1 * -1.

        ov_max = np.maximum(ov_max.max(), 0.01)
        bl_max = np.maximum(bl_max.max(), 0.01)

        overshoot = smoothing(overshoot)
        belowshoot = smoothing(belowshoot)



        maxx_ov = torch.max(overshoot) + 1e-12
        maxx_bl = torch.max(belowshoot) + 1e-12


        overshoot = overshoot / maxx_ov
        belowshoot = belowshoot / maxx_bl

        v_i = v_i - overshoot * ov_max + belowshoot * bl_max
        result_img = test_x + v_i


        overshoot = ((result_img - 1.) >= 0).type(torch.float32)
        belowshoot = ((result_img - 0.) <= 0).type(torch.float32)

        n_ov = overshoot.sum().item()
        n_bl = belowshoot.sum().item()
        n = n_ov + n_bl
        iter_i += 1

    print (iter_i, "iter")



    v_i = preprocess_channels(v_i, mean=[0., 0., 0.], std=std)



    return v_i

def preprocess_channels(x, mean, std):
    x_r = x[0:1, 0:1, :, :]
    x_g = x[0:1, 1:2, :, :]
    x_b = x[0:1, 2:3, :, :]
    x_r = (x_r - mean[0]) / std[0]
    x_g = (x_g - mean[1]) / std[1]
    x_b = (x_b - mean[2]) / std[2]

    return torch.cat((x_r, x_g, x_b), 1)

def deprocess_channels(x, mean, std):
    x_r = x[0:1, 0:1, :, :]
    x_g = x[0:1, 1:2, :, :]
    x_b = x[0:1, 2:3, :, :]
    x_r = x_r * std[0] + mean[0]
    x_g = x_g * std[1] + mean[1]
    x_b = x_b * std[2] + mean[2]

    return torch.cat((x_r, x_g, x_b), 1)