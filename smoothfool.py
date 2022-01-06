import torchvision.models as models
from PIL import Image
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable
import numpy as np
import torch
import math
import copy
import torchvision.transforms as transforms
import scipy.misc
import matplotlib.pyplot as plt
import os
import numbers
from torch.nn import functional as F
import torch.nn as nn
import argparse

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
t_max_r = (1. - mean[0]) / std[0]  # 2.248
t_max_g = (1. - mean[1]) / std[1]  # 2.428
t_max_b = (1. - mean[2]) / std[2]  # 2.640
t_min_r = (- mean[0]) / std[0]  # 2.248
t_min_g = (- mean[1]) / std[1]  # 2.428
t_min_b = (- mean[2]) / std[2]  # 2.640

labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')

# set random seed
torch.manual_seed(263)
np.random.seed(274)


def pred_cls(lbl):
    return labels[np.int(lbl)].split(',')[0]


########


#######


def deepfool(im, net, lambda_fac=2., num_classes=10, overshoot=0.02, max_iter=20, device='cuda'):
    image = copy.deepcopy(im)
    input_shape = image.size()

    f_image = net.forward(Variable(image, requires_grad=True)).data.cpu().numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]
    I = I[0:num_classes]
    label = I[0]

    pert_image = copy.deepcopy(image)
    r_tot = torch.zeros(input_shape).to(device)

    k_i = label
    loop_i = 0

    while k_i == label and loop_i < max_iter:

        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)

        pert = torch.Tensor([np.inf])[0].to(device)
        w = torch.zeros(input_shape).to(device)

        fs[0, I[0]].backward(retain_graph=True)

        grad_orig = copy.deepcopy(x.grad.data)

        for k in range(1, num_classes):
            zero_gradients(x)

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = copy.deepcopy(x.grad.data)

            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data

            pert_k = torch.abs(f_k) / w_k.norm()

            if pert_k < pert:
                pert = pert_k + 0.
                w = w_k + 0.

        r_i = torch.clamp(pert, min=1e-4) * w / w.norm()
        r_tot = r_tot + r_i

        pert_image = pert_image + r_i

        check_fool = image + (1 + overshoot) * r_tot
        k_i = torch.argmax(net.forward(Variable(check_fool, requires_grad=True)).data).item()

        loop_i += 1

    x = Variable(pert_image, requires_grad=True)
    fs = net.forward(x)
    (fs[0, k_i] - fs[0, label]).backward(retain_graph=True)
    grad = copy.deepcopy(x.grad.data)
    grad = grad / grad.norm()

    r_tot = lambda_fac * r_tot
    pert_image = image + r_tot

    return grad, pert_image, k_i


def smooth_clip(x, v, smoothing, max_iters=200):
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

    while n > 0 and iter_i < max_iters:
        result_img = test_x + v_i

        overshoot = ((result_img - 1.) >= 0).type(torch.float32)
        belowshoot = ((result_img - 0.) <= 0).type(torch.float32)

        ov_max = (result_img - 1.).data.cpu().numpy() * 0.1
        bl_max = (result_img - 0.).data.cpu().numpy() * 0.1 * -1.

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
    v_i = preprocess_channels(v_i, mean=[0., 0., 0.], std=std)
    return v_i, iter_i


def clip_value(x):
    xx = copy.deepcopy(x)
    x_0 = xx[0:1, :, :]
    x_1 = xx[1:2, :, :]
    x_2 = xx[2:3, :, :]
    x_0 = torch.clamp(x_0, t_min_r, t_max_r)
    x_1 = torch.clamp(x_1, t_min_g, t_max_g)
    x_2 = torch.clamp(x_2, t_min_b, t_max_b)
    x_c = torch.cat((x_0, x_1, x_2), 0)
    return x_c


def compute_roughness(r, smoothing):
    diff = r - smoothing(r)
    omega = torch.sum(diff ** 2)
    omega_normal = omega / torch.sum(r ** 2)
    return omega.item(), omega_normal.item()


def smoothfool(net, im, alpha_fac, dp_lambda, smoothing_func, max_iters=500, smooth_clipping=True, device='cuda'):
    net = net.to(device)
    im = im.to(device)
    x_i = copy.deepcopy(im).to(device)
    loop_i = 0
    f_image = net.forward(Variable(im[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
    label_nat = np.argmax(f_image)
    k_i = label_nat
    labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')
    total_clip_iters = 0
    attck_mon = []
    while loop_i < max_iters and k_i == label_nat:
        normal, x_adv, adv_lbl = deepfool(x_i[None, :, :, :], net, lambda_fac=dp_lambda, num_classes=10, device=device)
        normal_smooth = smoothing_func(normal)
        normal_smooth = normal_smooth / torch.norm(normal_smooth.view(-1))
        dot0 = torch.dot(normal.view(-1), x_adv.view(-1) - x_i.view(-1))
        dot1 = torch.dot(normal.view(-1), normal_smooth.view(-1))
        alpha = (dot0 / dot1) * alpha_fac
        normal_smooth = normal_smooth * alpha

        clip_iters = 0
        if smooth_clipping:
            normal_smooth, clip_iters = smooth_clip(x_i[None, :, :, :], normal_smooth, smoothing_func)
            if clip_iters > 198:
                print("clip_iters>iters_max")
                break
            total_clip_iters += clip_iters
            x_i = x_i + normal_smooth[0, :, :, :]
        else:
            x_i = clip_value(x_i + normal_smooth[0, ...])

        f_image = net.forward(Variable(x_i[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
        k_i = np.argmax(f_image)
        loop_i += 1
        print("         step: %03d, predicted label: %03d, prob of pred: %.3f, n of clip iters: %03d" % (
            loop_i, k_i, np.max(f_image), clip_iters))
        attck_mon.append(np.max(f_image))

        # track the performance of attack
        if len(attck_mon) > 10:
            del attck_mon[0]

    return x_i, loop_i, total_clip_iters, label_nat, k_i


def tensor2img(t):
    """
    converts the pytorch tensor to img by transposing the tensor and normalizing it
    :param t: input tensor
    :return: numpy array with last dim be the channels and all values in range [0, 1]
    """
    t_np = t.detach().cpu().numpy().transpose(1, 2, 0)
    t_np = (t_np - t_np.min()) / (t_np.max() - t_np.min())
    return t_np


############# EXP settings ##############################
alpha_fac = 1.1
dp_lambda = 1.5

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch implementation of SmoothFool')
    parser.add_argument('--sigma', default=5, type=float, help='smoothing factor')
    parser.add_argument('--type', default='uniform', type=str, help='type of smoothing')
    parser.add_argument('--smoothclip', default=False, type=bool,
                        help='clip adv samples using smoothclip or conventional clip')
    parser.add_argument('--net', default='resnet101', type=str,
                        help='network architecture to perform attack on, could be vgg16 or resent101')
    parser.add_argument('--img', default='./samples/ILSVRC2012_val_00000003.JPEG', type=str,
                        help='path to the input img')
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.net == 'vgg16':
        net = models.vgg16(pretrained=True)
    elif args.net == 'resnet101':
        net = models.resnet101(pretrained=True)
    else:
        raise ValueError('Network architecture is not defined!')

    # Switch to evaluation mode
    net.eval()

    smoothing = Smoothing(sig=args.sigma, type=args.type).to(device)

    # read the input image
    im_orig = Image.open(args.img)

    im = transforms.Compose([
        transforms.Scale(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])(im_orig)
    im = im.to(device)

    x_adv, loop_i, total_clip_iters, label_nat, label_adv = smoothfool(net, im, alpha_fac=alpha_fac,
                                                                       dp_lambda=dp_lambda,
                                                                       smoothing_func=smoothing,
                                                                       smooth_clipping=args.smoothclip,
                                                                       device=device)

    print("\nPredicted label for input sample: " + pred_cls(label_nat))
    print("Predicted label for adv sample: " + pred_cls(label_adv))

    # visualize the results

    from scipy.misc import imsave

    imsave('adv' + str(args.sigma) + '.png', tensor2img(x_adv))
    imsave('pert' + str(args.sigma) + '.png', tensor2img(x_adv-im))
    imsave('orig'+ '.png', tensor2img(im))


    plt.subplot(1, 3, 1)
    plt.title('Input sample')
    plt.imshow(tensor2img(im))
    plt.subplot(1, 3, 2)
    plt.title('Adv sample')
    plt.imshow(tensor2img(x_adv))
    plt.subplot(1, 3, 3)
    plt.title('Adv perturbation')
    plt.imshow(tensor2img(x_adv - im))
    plt.show()


