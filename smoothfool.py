from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable
from torch_utils import *
import copy
import os
import torchvision.transforms as transforms
import scipy.misc


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
t_max_r = (1. - mean[0]) / std[0]  # 2.248
t_max_g = (1. - mean[1]) / std[1]  # 2.428
t_max_b = (1. - mean[2]) / std[2]  # 2.640
t_min_r = (- mean[0]) / std[0]  # 2.248
t_min_g = (- mean[1]) / std[1]  # 2.428
t_min_b = (- mean[2]) / std[2]  # 2.640

labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')


def pred_cls(lbl):
   return labels[np.int(lbl)].split(',')[0]

def deepfool(im, net, lambda_fac=1.1, num_classes=10, overshoot=0.02, max_iter=20, device='cuda'):
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


def smoothfool(net, im, n_clusters=4, max_iters=50000, plot_cluters=False, device='cuda'):
    def clip_value(x):
        xx = copy.deepcopy(x)
        x_0 = xx[0:1, :, :]
        x_1 = xx[1:2, :, :]
        x_2 = xx[2:3, :, :]
        x_0 = torch.clamp(x_0, t_min_r, t_max_r)
        x_1 = torch.clamp(x_1, t_min_g, t_max_g)
        x_2 = torch.clamp(x_2, t_min_b, t_max_b)
        x_c = torch.cat((x_0, x_1, x_2), 0)
        error = torch.sum(torch.abs(x_c - xx))
        print ("clipping error [verify smoothclip] :", error.item())
        if error.item()>0.:
            exit('Error with smooth clipping')
        return x_c

    net = net.to(device)
    im = im.to(device)
    x_i = copy.deepcopy(im).to(device)
    loop_i = 0
    f_image = net.forward(Variable(im[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
    label = np.argmax(f_image)
    print (np.max(f_image))
    k_i = label

    smoothing = GaussianSmoothing(3, sigma=15.).to(device)#was 15 for lion figure
    labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')
    while loop_i<max_iters and k_i == label:
        normal, x_adv, adv_lbl = deepfool(x_i[None, :, :, :], net, 1., num_classes=20, device=device)
        normal_smooth = smoothing(normal)
        dot0 = torch.dot(normal.view(-1), x_adv.view(-1)-x_i.view(-1))
        dot1 = torch.dot(normal.view(-1), normal_smooth.view(-1))


        alpha = dot0/dot1
        print ("alpha:", alpha)

        cost = dot1/(torch.norm(normal.view(-1))*torch.norm(normal_smooth.view(-1)))
        print ("cost:", cost.item())

        normal_smooth = normal_smooth * alpha

        normal_smooth = smooth_clip_v2(x_i[None, :, :, :], normal_smooth, smoothing)

        x_i = x_i + normal_smooth[0, :, :, :]

        # verify smoothclip
        x_i = clip_value(x_i)
        f_image = net.forward(Variable(x_i[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
        label = np.argmax(f_image)
        loop_i += 1
        print ("step:", loop_i, "pred lbl:", pred_cls(label), "pred val:", np.max(f_image))
        print ("------------------")


    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    im_np = inv_tf(x_i.cpu().numpy().squeeze(), mean, std)
    im_np = np.clip(im_np, 0., 1.)

    im_orig = inv_tf(im.cpu().numpy().squeeze(), mean, std)

    tensor = im_np.transpose([2, 0, 1])
    tensor = torch.from_numpy(tensor).cuda()
    tensor= transforms.Compose([transforms.Normalize(mean=mean, std=std)])(tensor)
    f_image = net.forward(Variable(tensor[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
    label = np.argmax(f_image)

    print ("fool detail:", pred_cls(label), np.max(f_image), "<<<<<<<<<<<<<<<<<<<<<<")

    diff = x_i - im
    diff = diff.data.cpu().numpy().transpose([1, 2, 0])


    print ("diff norm2", np.linalg.norm(diff.reshape([-1])))
    print ("diff mean", np.mean(np.abs(diff)))
    print ("diff max", np.max(np.abs(diff)))




    diff = normalize(diff, p=True)

    plt.subplot(131)
    plt.imshow(im_orig)
    plt.subplot(132)
    plt.imshow(im_np)
    plt.subplot(133)
    plt.imshow(diff)

    plt.show()

    scipy.misc.imsave('orig.png', im_orig)
    scipy.misc.imsave('adv.png', im_np)
    scipy.misc.imsave('p.png', diff)

    exit()




    return x_i, label, k_i

