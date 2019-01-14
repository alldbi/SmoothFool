from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable
from torch_utils import *
from np_utils import *

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
t_max_r = (1. - mean[0]) / std[0]  # 2.248
t_max_g = (1. - mean[1]) / std[1]  # 2.428
t_max_b = (1. - mean[2]) / std[2]  # 2.640
t_min_r = (- mean[0]) / std[0]  # 2.248
t_min_g = (- mean[1]) / std[1]  # 2.428
t_min_b = (- mean[2]) / std[2]  # 2.640

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



