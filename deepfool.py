import numpy as np
import torch
import copy
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable
from torch_utils import *
from np_utils import  *
import matplotlib.pyplot as plt

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
t_max_r = (1. - mean[0]) / std[0]  # 2.248
t_max_g = (1. - mean[1]) / std[1]  # 2.428
t_max_b = (1. - mean[2]) / std[2]  # 2.640
t_min_r = (- mean[0]) / std[0]  # 2.248
t_min_g = (- mean[1]) / std[1]  # 2.428
t_min_b = (- mean[2]) / std[2]  # 2.640

def deepfool(im, net, lambda_fac=1., num_classes=50, overshoot=0.02, max_iter=20, device='cuda', smoothing=None):
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
        var_min = 1000.
        cost_max = 0.
        for k in range(1, num_classes):
            zero_gradients(x)

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = copy.deepcopy(x.grad.data)

            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data

            pert_k = torch.abs(f_k) / w_k.norm()

            # w_kk = smoothing(w_k)

            # cos_t = torch.dot(w_k.view(-1), w_kk.view(-1))
            # mag = torch.norm(w_k.view(-1), 2)*torch.norm(w_kk.view(-1), 2)
            # cos_t = cos_t / mag



            var = w_k.var()

            # if var < var_min:
            if pert_k < pert:
            # if cos_t>cost_max:
            #     cost_max=cos_t
                var_min = var
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


    # fshow = r_tot.data.cpu().numpy().squeeze()
    # print fshow.var(), "variance"
    # print torch.norm(r_tot.view(-1),2), "norm2"
    # fshow = normalize(fshow)
    # fshow = np.transpose(fshow, [1, 2, 0])
    # plt.imshow(fshow)
    # plt.show()



    # r_tot = lambda_fac * r_tot
    pert_image = image + r_tot

    return grad, pert_image, k_i



def deepfool_lp(im, net, lambda_fac=1., num_classes=50, overshoot=0.02, max_iter=200, device='cuda'):
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

    smoothing = GaussianSmoothing(3, 200, 20.).to(device)
    def clip_value(x):
        xx = copy.deepcopy(x)
        x_0 = xx[0:1, :, :]
        x_1 = xx[1:2, :, :]
        x_2 = xx[2:3, :, :]
        x_0 = torch.clamp(x_0, -mean[0]/std[0], (1.-mean[0])/std[0])
        x_1 = torch.clamp(x_1, -mean[1]/std[1], (1.-mean[1])/std[1])
        x_2 = torch.clamp(x_2, -mean[2]/std[2], (1.-mean[2])/std[2])
        return torch.cat((x_0, x_1, x_2), 0)
    while k_i == label and loop_i < max_iter:

        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)

        pert = torch.Tensor([np.inf])[0].to(device)
        w = torch.zeros(input_shape).to(device)

        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = copy.deepcopy(x.grad.data)
        var_min = 1000.
        for k in range(1, num_classes):
            zero_gradients(x)

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = copy.deepcopy(x.grad.data)

            w_k = cur_grad - grad_orig

            w_kk = smoothing(w_k)

            f_k = (fs[0, I[k]] - fs[0, I[0]]).data

            pert_k = torch.abs(f_k) / w_kk.norm()
            # print pert_k
            # pert_k = (1.-torch.acos(cos_t)/3.15) * torch.abs(f_k) / w_kk.norm()


            var = w_k.var()

            # if var < var_min:
            if pert_k < pert:
                # var_min = var
                pert = pert_k + 0.
                w = w_kk + 0.
        print (pert)
        r_i = torch.clamp(pert, min=1e-4) * w / w.norm()*0.1

        print (r_i.size())
        ri_r = r_i[0, 0, :, :].data.cpu().numpy()
        ri_g = r_i[0, 1, :, :].data.cpu().numpy()
        ri_b = r_i[0, 2, :, :].data.cpu().numpy()

        # plt.subplot(131)
        # plt.imshow(normalize(ri_r))
        # plt.subplot(132)
        # plt.imshow(normalize(ri_g))
        # plt.subplot(133)
        # plt.imshow(normalize(ri_b))
        # plt.show()
        #
        #
        # print ri_r.min(), ri_r.max()
        # print ri_g.min(), ri_g.max()
        # print ri_b.min(), ri_b.max()
        # exit()


        # print r_i.data.cpu().numpy().min()
        # print r_i.data.cpu().numpy().max(), "max"

        # r_i = torch.clamp(r_i, -0.05, 0.05)

        # r_i = smoothing(r_i)
        # fshow = r_i.data.cpu().numpy().squeeze()
        # fshow = normalize(fshow)
        # fshow = np.transpose(fshow, [1, 2, 0])
        # plt.imshow(fshow)
        # plt.show()





        # find the ratio that prevents from sta



        test_img = copy.deepcopy(pert_image)
        result_img = test_img + r_i
        result_img_r = result_img[0, 0, :, :].view(-1)
        result_img_g = result_img[0, 1, :, :].view(-1)
        result_img_b = result_img[0, 2, :, :].view(-1)


        t_max_r = (1.-mean[0])/std[0] # 2.248
        t_max_g = (1.-mean[1])/std[1] # 2.428
        t_max_b = (1.-mean[2])/std[2] # 2.640

        # print t_max_r, t_max_g, t_max_b

        overshoot_r = torch.max(result_img_r - t_max_r).item()
        overshoot_g = torch.max(result_img_g - t_max_g).item()
        overshoot_b = torch.max(result_img_b - t_max_b).item()
        overshoot_r = (np.maximum(overshoot_r, 0.)==0.).astype(np.float32)
        overshoot_g = (np.maximum(overshoot_g, 0.)==0.).astype(np.float32)
        overshoot_b = (np.maximum(overshoot_b, 0.)==0.).astype(np.float32)
        print (overshoot_r, overshoot_b, overshoot_g, "overshoots")

        pert_image[0, 0, :, :] = pert_image[0, 0, :, :] + r_i[0, 0, :, :]# * overshoot_r
        pert_image[0, 1, :, :] = pert_image[0, 1, :, :] + r_i[0, 1, :, :]# * overshoot_g
        pert_image[0, 2, :, :] = pert_image[0, 2, :, :] + r_i[0, 2, :, :]# * overshoot_b


        r_tot[0, 0, :, :] = r_tot[0, 0, :, :] + r_i[0, 0, :, :]# * overshoot_r
        r_tot[0, 1, :, :] = r_tot[0, 1, :, :] + r_i[0, 1, :, :]# * overshoot_g
        r_tot[0, 2, :, :] = r_tot[0, 2, :, :] + r_i[0, 2, :, :]# * overshoot_b

        # argmax_r = overshoot_r.argmax()
        # argmax_g = overshoot_g.argmax()
        # argmax_b = overshoot_b.argmax()
        # print source_img_g[argmax_g]
        # print r_i_g[argmax_g]
        # print overshoot_g.max(), "overshoot"
        # exit()



        # r_tot = r_tot + r_i


        # r_tot = torch.clamp(r_tot, -0.5, 0.5)

        # pert_image = image + r_tot










        # pert_image = pert_image + r_i
        pert_image = clip_value(pert_image)





        check_fool = image + (1. + overshoot) * r_tot
        # check_fool = clip_value(check_fool)
        k_i = torch.argmax(net.forward(Variable(check_fool, requires_grad=True)).data).item()
        print (loop_i, torch.max(net.forward(Variable(check_fool, requires_grad=True)).data).item())

        loop_i += 1

    x = Variable(pert_image, requires_grad=True)
    fs = net.forward(x)
    (fs[0, k_i] - fs[0, label]).backward(retain_graph=True)
    grad = copy.deepcopy(x.grad.data)
    grad = grad / grad.norm()

    fshow = r_tot.data.cpu().numpy().squeeze()
    print (fshow.var(), "variance")
    print (torch.norm(r_tot.view(-1), 2), "norm2")
    fshow = normalize(fshow)
    fshow = np.transpose(fshow, [1, 2, 0])
    # plt.imshow(fshow)
    # plt.show()

    # r_tot = lambda_fac * r_tot
    # pert_image = image + r_tot
    # pert_image = clip_value(pert_image)



    return grad, pert_image, r_tot, label, k_i, loop_i


def deepfool_lp_overshoot(im, net, lambda_fac=1., num_classes=50, overshoot=0.02, max_iter=200, device='cuda'):
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

    # smoothing = GaussianSmoothing(3, 128, 10.).to(device)
    smoothing = GaussianSmoothing(3, 200, 20.).to(device)
    def clip_value(x):
        xx = copy.deepcopy(x)
        x_0 = xx[0:1, :, :]
        x_1 = xx[1:2, :, :]
        x_2 = xx[2:3, :, :]
        x_0 = torch.clamp(x_0, -mean[0]/std[0], (1.-mean[0])/std[0])
        x_1 = torch.clamp(x_1, -mean[1]/std[1], (1.-mean[1])/std[1])
        x_2 = torch.clamp(x_2, -mean[2]/std[2], (1.-mean[2])/std[2])
        return torch.cat((x_0, x_1, x_2), 0)
    while k_i == label and loop_i < max_iter:

        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)

        pert = torch.Tensor([np.inf])[0].to(device)
        w = torch.zeros(input_shape).to(device)

        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = copy.deepcopy(x.grad.data)
        var_min = 1000.
        for k in range(1, num_classes):
            zero_gradients(x)

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = copy.deepcopy(x.grad.data)

            w_k = cur_grad - grad_orig

            w_kk = smoothing(w_k)

            cos_t = torch.dot(w_k.view(-1), w_kk.view(-1))
            mag = torch.norm(w_k.view(-1), 2)*torch.norm(w_kk.view(-1), 2)
            cos_t = cos_t / mag
            # print cos_t, "cos_t"



            # input = F.pad(input, (2, 2, 2, 2), mode='reflect')



            f_k = (fs[0, I[k]] - fs[0, I[0]]).data

            pert_k = torch.abs(f_k) / w_kk.norm()
            # print pert_k
            # pert_k = (1.-torch.acos(cos_t)/3.15) * torch.abs(f_k) / w_kk.norm()


            var = w_k.var()

            # if var < var_min:
            if pert_k < pert:
                # var_min = var
                pert = pert_k + 0.
                w = w_kk + 0.
        r_i = torch.clamp(pert, min=1e-4) * w / w.norm()*0.1

        # find the ratio that prevents from sta

        n=1
        t_max_r = (1. - mean[0]) / std[0]  # 2.248
        t_max_g = (1. - mean[1]) / std[1]  # 2.428
        t_max_b = (1. - mean[2]) / std[2]  # 2.640
        t_min_r = (- mean[0]) / std[0]  # 2.248
        t_min_g = (- mean[1]) / std[1]  # 2.428
        t_min_b = (- mean[2]) / std[2]  # 2.640
        while n>0:

            test_img = copy.deepcopy(pert_image)
            result_img = test_img + r_i
            result_img_r = result_img[0, 0, :, :]
            result_img_g = result_img[0, 1, :, :]
            result_img_b = result_img[0, 2, :, :]

            # print t_max_r, t_max_g, t_max_b

            overshoot_r = ((result_img_r - t_max_r)>=0).type(torch.float32).view([1, 1, 224, 224])
            overshoot_g = ((result_img_g - t_max_g)>=0).type(torch.float32).view([1, 1, 224, 224])
            overshoot_b = ((result_img_b - t_max_b)>=0).type(torch.float32).view([1, 1, 224, 224])

            belowshoot_r = ((result_img_r - t_min_r)<=0).type(torch.float32).view([1, 1, 224, 224])
            belowshoot_g = ((result_img_g - t_min_g)<=0).type(torch.float32).view([1, 1, 224, 224])
            belowshoot_b = ((result_img_b - t_min_b)<=0).type(torch.float32).view([1, 1, 224, 224])


            # plt.subplot(121)
            # plt.imshow(overshoot_r[0, 0, :, :].data.cpu().numpy())
            # plt.subplot(122)
            # plt.imshow(belowshoot_r[0, 0, :, :].data.cpu().numpy())
            # plt.show()



            ov_r_max = (result_img_r-t_max_r).data.cpu().numpy()
            ov_g_max = (result_img_g-t_max_g).data.cpu().numpy()
            ov_b_max = (result_img_b-t_max_b).data.cpu().numpy()

            bl_r_max = (result_img_r-t_min_r).data.cpu().numpy()*-1.
            bl_g_max = (result_img_g-t_min_g).data.cpu().numpy()*-1.
            bl_b_max = (result_img_b-t_min_b).data.cpu().numpy()*-1.


            ov_r_max = np.maximum(ov_r_max.max(), 0.01)
            ov_g_max = np.maximum(ov_g_max.max(), 0.01)
            ov_b_max = np.maximum(ov_b_max.max(), 0.01)

            bl_r_max = np.maximum(bl_r_max.max(), 0.01)
            bl_g_max = np.maximum(bl_g_max.max(), 0.01)
            bl_b_max = np.maximum(bl_b_max.max(), 0.01)





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
            # # print overshoot_r.shape
            # plt.subplot(131)
            # plt.imshow(overshoot_r[0, 0, :, :])
            # plt.subplot(132)
            # plt.imshow(overshoot_g[0, 0, :, :])
            # plt.subplot(133)
            # plt.imshow(overshoot_b[0, 0, :, :])
            # plt.show()
            # exit()

            maxx_ov = torch.max(overshoot_r.max(), torch.max(overshoot_b.max(), overshoot_g.max()))+1e-12
            maxx_bl = torch.max(belowshoot_r.max(), torch.max(belowshoot_g.max(), belowshoot_b.max()))+1e-12

            # print maxx, overshoot_r.max(), overshoot_g.max(), overshoot_b.max()

            overshoot_r = overshoot_r/maxx_ov #(overshoot_r>0.).type(torch.float32)
            overshoot_g = overshoot_g/maxx_ov #(overshoot_r>0.).type(torch.float32)
            overshoot_b = overshoot_b/maxx_ov #(overshoot_r>0.).type(torch.float32)

            belowshoot_r = belowshoot_r/maxx_bl
            belowshoot_g = belowshoot_g/maxx_bl
            belowshoot_b = belowshoot_b/maxx_bl


            # plt.subplot(121)
            # plt.imshow((overshoot_r[0, 0, :, :].data.cpu().numpy()))


            # overshoot_r = (overshoot_r>0.02).type(torch.float32)
            # overshoot_g = (overshoot_g>0.02).type(torch.float32)
            # overshoot_b = (overshoot_b>0.02).type(torch.float32)


            # plt.subplot(122)
            # plt.imshow((overshoot_r[0, 0, :, :].data.cpu().numpy()))
            # plt.show()



            rr = copy.deepcopy(r_i)
            r_i[0, 0, :, :] = r_i[0, 0, :, :] - overshoot_r[0, 0, :, :]*ov_r_max + belowshoot_r[0, 0, :, :]*bl_r_max
            r_i[0, 1, :, :] = r_i[0, 1, :, :] - overshoot_g[0, 0, :, :]*ov_g_max + belowshoot_g[0, 0, :, :]*bl_g_max
            r_i[0, 2, :, :] = r_i[0, 2, :, :] - overshoot_b[0, 0, :, :]*ov_b_max + belowshoot_b[0, 0, :, :]*bl_b_max
            result_img = test_img + r_i
            result_img_r = result_img[0, 0, :, :]
            result_img_g = result_img[0, 1, :, :]
            result_img_b = result_img[0, 2, :, :]

            # print t_max_r, t_max_g, t_max_b

            overshoot_r = ((result_img_r - t_max_r) >= 0).type(torch.float32)
            overshoot_g = ((result_img_g - t_max_g) >= 0).type(torch.float32)
            overshoot_b = ((result_img_b - t_max_b) >= 0).type(torch.float32)

            belowshoot_r = ((result_img_r - t_min_r)<=0).type(torch.float32)
            belowshoot_g = ((result_img_g - t_min_g)<=0).type(torch.float32)
            belowshoot_b = ((result_img_b - t_min_b)<=0).type(torch.float32)


            n_ov_r = overshoot_r.sum().item()
            n_ov_g = overshoot_g.sum().item()
            n_ov_b = overshoot_b.sum().item()

            n_bl_r = belowshoot_r.sum().item()
            n_bl_g = belowshoot_g.sum().item()
            n_bl_b = belowshoot_b.sum().item()




            # print "n_ov_r:", n_ov_r, "n_ov_g:", n_ov_g, "n_ov_b", n_ov_b, "n_ov_total:", n_ov_r+n_ov_g+n_ov_b
            # print "n_bl_r:", n_bl_r, "n_bl_g:", n_bl_g, "n_bl_b", n_bl_b, "n_bl_total:", n_bl_r+n_bl_g+n_bl_b


            n = n_ov_r + n_ov_g + n_ov_b + n_bl_r + n_bl_g + n_bl_b
            # print "total:", n




        #
        # plt.subplot(131)
        # plt.imshow((rr[0, 0, :, :].data.cpu().numpy()))
        # plt.subplot(132)
        # plt.imshow((r_i[0, 0, :, :].data.cpu().numpy()))
        # plt.subplot(133)
        # plt.imshow((overshoot_r[0, 0, :, :].data.cpu().numpy()))
        # plt.show()
        # exit()

        pert_image[0, 0, :, :] = pert_image[0, 0, :, :] + r_i[0, 0, :, :]
        pert_image[0, 1, :, :] = pert_image[0, 1, :, :] + r_i[0, 1, :, :]
        pert_image[0, 2, :, :] = pert_image[0, 2, :, :] + r_i[0, 2, :, :]



        pimg = pert_image[0, :, :, :].data.cpu().numpy()

        # print pimg.min(), pimg.max()
        # exit()


        pimg = normalize(pimg)
        pimg = pimg.transpose([1, 2, 0])

        # print pimg.11max(), pimg.min()

        # plt.imshow(pimg)
        # plt.show()



        r_tot[0, 0, :, :] = r_tot[0, 0, :, :] + r_i[0, 0, :, :]
        r_tot[0, 1, :, :] = r_tot[0, 1, :, :] + r_i[0, 1, :, :]
        r_tot[0, 2, :, :] = r_tot[0, 2, :, :] + r_i[0, 2, :, :]


        # pert_image = clip_value(pert_image)
        overshoot_r = ((pert_image[0, 0, :, :] - t_max_r)>0).type(torch.float32).view([1, 1, 224, 224])
        overshoot_g = ((pert_image[0, 1, :, :] - t_max_g)>0).type(torch.float32).view([1, 1, 224, 224])
        overshoot_b = ((pert_image[0, 2, :, :] - t_max_b)>0).type(torch.float32).view([1, 1, 224, 224])
        ov1 = overshoot_r.data.cpu().numpy()
        ov2 = overshoot_g.data.cpu().numpy()
        ov3 = overshoot_b.data.cpu().numpy()

        belowshoot_r = ((pert_image[0, 0, :, :] - t_min_r)<0).type(torch.float32).view([1, 1, 224, 224])
        belowshoot_g = ((pert_image[0, 1, :, :] - t_min_g)<0).type(torch.float32).view([1, 1, 224, 224])
        belowshoot_b = ((pert_image[0, 2, :, :] - t_min_b)<0).type(torch.float32).view([1, 1, 224, 224])
        ov1 = overshoot_r.data.cpu().numpy()
        ov2 = overshoot_g.data.cpu().numpy()
        ov3 = overshoot_b.data.cpu().numpy()
        bl1 = belowshoot_r.data.cpu().numpy()
        bl2 = belowshoot_g.data.cpu().numpy()
        bl3 = belowshoot_b.data.cpu().numpy()

        ov1 = np.where(ov1==1)
        ov2 = np.where(ov2==1)
        ov3 = np.where(ov3==1)
        bl1 = np.where(bl1==1)
        bl2 = np.where(bl2==1)
        bl3 = np.where(bl3==1)


        print (ov1[0].shape[0],ov2[0].shape[0], ov3[0].shape[0], "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        print (bl1[0].shape[0],bl2[0].shape[0], bl3[0].shape[0], "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")


        # plt.imshow(normalize(ov))
        # plt.show()

        check_fool = image + (1. + 0.) * r_tot
        # check_fool = clip_value(check_fool)
        k_i = torch.argmax(net.forward(Variable(check_fool, requires_grad=True)).data).item()
        print (loop_i, torch.max(net.forward(Variable(check_fool, requires_grad=True)).data).item())





        loop_i += 1

    x = Variable(pert_image, requires_grad=True)
    fs = net.forward(x)
    (fs[0, k_i] - fs[0, label]).backward(retain_graph=True)
    grad = copy.deepcopy(x.grad.data)
    grad = grad / grad.norm()

    fshow = r_tot.data.cpu().numpy().squeeze()
    print (fshow.var(), "variance")
    print (torch.norm(r_tot.view(-1), 2), "norm2")
    fshow = normalize(fshow)
    fshow = np.transpose(fshow, [1, 2, 0])
    # plt.imshow(fshow)
    # plt.show()

    # r_tot = lambda_fac * r_tot
    # pert_image = image + r_tot
    # pert_image = clip_value(pert_image)



    return grad, pert_image, r_tot, label, k_i, loop_i

def deepfool_lp_overshoot2(im, net, lambda_fac=1., num_classes=50, overshoot=0.02, max_iter=500, device='cuda'):
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

    # smoothing = GaussianSmoothing(3, 128, 10.).to(device)
    smoothing = GaussianSmoothing(3, 200, 20.).to(device)
    def clip_value(x):
        xx = copy.deepcopy(x)

        x_0 = xx[0:1, 0:1, :, :]
        x_1 = xx[0:1, 1:2, :, :]
        x_2 = xx[0:1, 2:3, :, :]



        # print t_min_r, t_min_g, t_min_b
        # print t_max_r, t_max_g, t_max_b
        #
        #
        #
        # print "x0:", torch.min(x_0).item(), torch.max(x_0).item()
        # print "x1:", torch.min(x_1).item(), torch.max(x_1).item()
        # print "x2:", torch.min(x_2).item(), torch.max(x_2).item()
        # exit()


        x_0 = torch.clamp(x_0, t_min_r, t_max_r)
        x_1 = torch.clamp(x_1, t_min_g, t_max_g)
        x_2 = torch.clamp(x_2, t_min_b, t_max_b)


        x_c = torch.cat((x_0, x_1, x_2), 1)

        error = torch.sum(torch.abs(x_c - xx))
        print (error.item(), "errooooorrr")
        return x_c
    while k_i == label and loop_i < max_iter:

        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)

        pert = torch.Tensor([np.inf])[0].to(device)
        w = torch.zeros(input_shape).to(device)

        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = copy.deepcopy(x.grad.data)
        var_min = 1000.
        cost_t_max = 0.
        for k in range(1, num_classes):
            zero_gradients(x)

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = copy.deepcopy(x.grad.data)

            w_k = cur_grad - grad_orig

            w_kk = smoothing(w_k)

            cos_t = torch.dot(w_k.view(-1), w_kk.view(-1))
            mag = torch.norm(w_k.view(-1), 2)*torch.norm(w_kk.view(-1), 2)
            cos_t = cos_t / mag
            # print "cos_t", cos_t



            # input = F.pad(input, (2, 2, 2, 2), mode='reflect')



            f_k = (fs[0, I[k]] - fs[0, I[0]]).data

            pert_k = torch.abs(f_k) / w_kk.norm()
            # print pert_k
            # pert_k = (1.-torch.acos(cos_t)/3.15) * torch.abs(f_k) / w_kk.norm()


            var = w_k.var()

            # if var < var_min:
            # if pert_k < pert:
            if cost_t_max<cos_t:
                # var_min = var
                cost_t_max = cos_t
                pert = pert_k + 0.
                w = w_kk + 0.
        r_i = torch.clamp(pert, min=1e-4) * w / w.norm() * 0.1

        # find the ratio that prevents from sta

        r_i = smooth_clip(pert_image, r_i, smoothing)


        pert_image[0, 0, :, :] = pert_image[0, 0, :, :] + r_i[0, 0, :, :]
        pert_image[0, 1, :, :] = pert_image[0, 1, :, :] + r_i[0, 1, :, :]
        pert_image[0, 2, :, :] = pert_image[0, 2, :, :] + r_i[0, 2, :, :]



        pimg = pert_image[0, :, :, :].data.cpu().numpy()

        # print pimg.min(), pimg.max()
        # exit()


        pimg = normalize(pimg)
        pimg = pimg.transpose([1, 2, 0])

        # print pimg.max(), pimg.min()

        # plt.imshow(pimg)
        # plt.show()



        r_tot[0, 0, :, :] = r_tot[0, 0, :, :] + r_i[0, 0, :, :]
        r_tot[0, 1, :, :] = r_tot[0, 1, :, :] + r_i[0, 1, :, :]
        r_tot[0, 2, :, :] = r_tot[0, 2, :, :] + r_i[0, 2, :, :]



        # overshoot_r = ((pert_image[0, 0, :, :] - t_max_r)>0).type(torch.float32).view([1, 1, 224, 224])
        # overshoot_g = ((pert_image[0, 1, :, :] - t_max_g)>0).type(torch.float32).view([1, 1, 224, 224])
        # overshoot_b = ((pert_image[0, 2, :, :] - t_max_b)>0).type(torch.float32).view([1, 1, 224, 224])
        # ov1 = overshoot_r.data.cpu().numpy()
        # ov2 = overshoot_g.data.cpu().numpy()
        # ov3 = overshoot_b.data.cpu().numpy()
        #
        # belowshoot_r = ((pert_image[0, 0, :, :] - t_min_r)<0).type(torch.float32).view([1, 1, 224, 224])
        # belowshoot_g = ((pert_image[0, 1, :, :] - t_min_g)<0).type(torch.float32).view([1, 1, 224, 224])
        # belowshoot_b = ((pert_image[0, 2, :, :] - t_min_b)<0).type(torch.float32).view([1, 1, 224, 224])
        # ov1 = overshoot_r.data.cpu().numpy()
        # ov2 = overshoot_g.data.cpu().numpy()
        # ov3 = overshoot_b.data.cpu().numpy()
        # bl1 = belowshoot_r.data.cpu().numpy()
        # bl2 = belowshoot_g.data.cpu().numpy()
        # bl3 = belowshoot_b.data.cpu().numpy()
        #
        # ov1 = np.where(ov1==1)
        # ov2 = np.where(ov2==1)
        # ov3 = np.where(ov3==1)
        # bl1 = np.where(bl1==1)
        # bl2 = np.where(bl2==1)
        # bl3 = np.where(bl3==1)

        pert_image = clip_value(pert_image)
        # print ov1[0].shape[0],ov2[0].shape[0], ov3[0].shape[0], "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
        # print bl1[0].shape[0],bl2[0].shape[0], bl3[0].shape[0], "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"


        # plt.imshow(normalize(ov))
        # plt.show()

        check_fool = image + (1. + 0.) * r_tot
        # check_fool = clip_value(check_fool)
        k_i = torch.argmax(net.forward(Variable(check_fool, requires_grad=True)).data).item()
        print (loop_i, torch.max(net.forward(Variable(check_fool, requires_grad=True)).data).item())





        loop_i += 1

    x = Variable(pert_image, requires_grad=True)
    fs = net.forward(x)
    (fs[0, k_i] - fs[0, label]).backward(retain_graph=True)
    grad = copy.deepcopy(x.grad.data)
    grad = grad / grad.norm()


    diff = torch.mean(torch.abs(pert_image - im)).item()
    print ("mean of difference: ", diff)
    max_diff = torch.max(torch.abs(pert_image-im)).item()
    print ("max of difference: ", max_diff)
    # fshow = r_tot.data.cpu().numpy().squeeze()
    # print fshow.var(), "variance"
    # print torch.norm(r_tot.view(-1), 2), "norm2"
    # fshow = normalize(fshow)
    # fshow = np.transpose(fshow, [1, 2, 0])
    # plt.imshow(fshow)
    # plt.show()

    # r_tot = lambda_fac * r_tot
    # pert_image = image + r_tot
    # pert_image = clip_value(pert_image)



    return grad, pert_image, r_tot, label, k_i, loop_i

def deepfool_lp2(im, net, lambda_fac=1., num_classes=50, overshoot=0.02, max_iter=200, device='cuda'):
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

    smoothing = GaussianSmoothing(3, 128, 10).to(device)
    def clip_value(x):


        xx = copy.deepcopy(x)
        x_0 = xx[0:1, :, :]
        x_1 = xx[1:2, :, :]
        x_2 = xx[2:3, :, :]
        x_0 = torch.clamp(x_0, -mean[0]/std[0], (1.-mean[0])/std[0])
        x_1 = torch.clamp(x_1, -mean[1]/std[1], (1.-mean[1])/std[1])
        x_2 = torch.clamp(x_2, -mean[2]/std[2], (1.-mean[2])/std[2])
        return torch.cat((x_0, x_1, x_2), 0)
    while k_i == label and loop_i < max_iter:

        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)

        pert = torch.Tensor([np.inf])[0].to(device)
        f_kkk = torch.Tensor([np.inf])[0].to(device)
        w = torch.zeros(input_shape).to(device)

        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = copy.deepcopy(x.grad.data)
        var_min = 1000.
        for k in range(1, num_classes):
            zero_gradients(x)

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = copy.deepcopy(x.grad.data)

            w_k = cur_grad - grad_orig
            w_kk = smoothing(w_k)
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data

            pert_k = torch.abs(f_k) / w_kk.norm()
            # print pert_k
            # pert_k = (1.-torch.acos(cos_t)/3.15) * torch.abs(f_k) / w_kk.norm()


            var = w_k.var()

            # if var < var_min:
            if pert_k < pert:
                # var_min = var
                pert = pert_k + 0.
                w = w_k + 0.
                f_kkk = f_k




        # w_np = w_k.data.cpu().numpy()
        #
        # np.save('w', w_np)
        #
        #
        # w_np = w_np[0, 0, :, :]
        # w_np = normalize(w_np)
        # plt.imshow(w_np)
        # plt.show()
        # exit()

        n_overshoot = 300
        step = 0
        test_img = copy.deepcopy(pert_image)
        while n_overshoot>200:
            print ("step:", step, n_overshoot)
            w_smooth = smoothing(w)

            pert = torch.abs(f_kkk) / w_smooth.norm()


            r_i = torch.clamp(pert, min=1e-4) * w_smooth / w_smooth.norm()*0.1
            result_img = test_img + r_i
            t_max_r = (1.-mean[0])/std[0] # 2.248
            t_max_g = (1.-mean[1])/std[1] # 2.428
            t_max_b = (1.-mean[2])/std[2] # 2.640

            threshold = torch.zeros([1, 3, 224, 224]).cuda()
            threshold[0, 0, :, :] = t_max_r*0.8
            threshold[0, 1, :, :] = t_max_g*0.8
            threshold[0, 2, :, :] = t_max_b*0.8

            diff = (result_img <= threshold).type(torch.float32)
            n_overshoot = 224*224*3 - torch.sum(diff)
            w = w * diff
            step = step + 1






        pert_image = pert_image + r_i
        r_tot = r_tot + r_i


        # overshoot = result_img
        #
        # # find the ratio that prevents from sta
        #
        #
        #
        # test_img = copy.deepcopy(pert_image)
        # result_img = test_img + r_i
        # result_img_r = result_img[0, 0, :, :].view(-1)
        # result_img_g = result_img[0, 1, :, :].view(-1)
        # result_img_b = result_img[0, 2, :, :].view(-1)
        #
        # source_img_r = pert_image[0, 0, :, :].view(-1)
        # source_img_g = pert_image[0, 1, :, :].view(-1)
        # r_i_r = r_i[0, 0, :, :].view(-1)
        # r_i_g = r_i[0, 1, :, :].view(-1)
        #
        # t_max_r = (1.-mean[0])/std[0] # 2.248
        # t_max_g = (1.-mean[1])/std[1] # 2.428
        # t_max_b = (1.-mean[2])/std[2] # 2.640
        #
        # # print t_max_r, t_max_g, t_max_b
        #
        # overshoot_r = torch.max(result_img_r - t_max_r).item()
        # overshoot_g = torch.max(result_img_g - t_max_g).item()
        # overshoot_b = torch.max(result_img_b - t_max_b).item()
        # overshoot_r = (np.maximum(overshoot_r, 0.)==0.).astype(np.float32)
        # overshoot_g = (np.maximum(overshoot_g, 0.)==0.).astype(np.float32)
        # overshoot_b = (np.maximum(overshoot_b, 0.)==0.).astype(np.float32)
        # print overshoot_r, overshoot_b, overshoot_g
        # # print overshoot_r
        # # print overshoot_b
        # # print overshoot_g
        # # exit()
        # #
        # #
        # # print r_i.size()
        # # print pert_image.size()
        #
        # pert_image[0, 0, :, :] = pert_image[0, 0, :, :] + overshoot_r*r_i[0, 0, :, :]
        # pert_image[0, 1, :, :] = pert_image[0, 1, :, :] + overshoot_g*r_i[0, 1, :, :]
        # pert_image[0, 2, :, :] = pert_image[0, 2, :, :] + overshoot_b*r_i[0, 2, :, :]
        #
        #
        # r_tot[0, 0, :, :] = r_tot[0, 0, :, :] + overshoot_r*r_i[0, 0, :, :]
        # r_tot[0, 1, :, :] = r_tot[0, 1, :, :] + overshoot_g*r_i[0, 1, :, :]
        # r_tot[0, 2, :, :] = r_tot[0, 2, :, :] + overshoot_b*r_i[0, 2, :, :]

        # argmax_r = overshoot_r.argmax()
        # argmax_g = overshoot_g.argmax()
        # argmax_b = overshoot_b.argmax()
        # print source_img_g[argmax_g]
        # print r_i_g[argmax_g]
        # print overshoot_g.max(), "overshoot"
        # exit()



        # r_tot = r_tot + r_i


        # r_tot = torch.clamp(r_tot, -0.5, 0.5)

        # pert_image = image + r_tot










        # pert_image = pert_image + r_i
        # pert_image = clip_value(pert_image)





        check_fool = image + (1. + overshoot) * r_tot
        # check_fool = clip_value(check_fool)
        k_i = torch.argmax(net.forward(Variable(check_fool, requires_grad=True)).data).item()
        print (torch.max(net.forward(Variable(check_fool, requires_grad=True)).data).item())

        loop_i += 1

    x = Variable(pert_image, requires_grad=True)
    fs = net.forward(x)
    (fs[0, k_i] - fs[0, label]).backward(retain_graph=True)
    grad = copy.deepcopy(x.grad.data)
    grad = grad / grad.norm()

    fshow = r_tot.data.cpu().numpy().squeeze()
    print (fshow.var(), "variance")
    print (torch.norm(r_tot.view(-1), 2), "norm2")
    fshow = normalize(fshow)
    fshow = np.transpose(fshow, [1, 2, 0])
    # plt.imshow(fshow)
    # plt.show()

    # r_tot = lambda_fac * r_tot
    # pert_image = image + r_tot
    # pert_image = clip_value(pert_image)



    return grad, pert_image, r_tot, label, k_i, loop_i


def deepfool_lp3(im, net, lambda_fac=1., num_classes=50, overshoot=0.02, max_iter=200, device='cuda'):
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

    smoothing = GaussianSmoothing(3, 128, 10).to(device)
    def clip_value(x):


        xx = copy.deepcopy(x)
        x_0 = xx[0:1, :, :]
        x_1 = xx[1:2, :, :]
        x_2 = xx[2:3, :, :]
        x_0 = torch.clamp(x_0, -mean[0]/std[0], (1.-mean[0])/std[0])
        x_1 = torch.clamp(x_1, -mean[1]/std[1], (1.-mean[1])/std[1])
        x_2 = torch.clamp(x_2, -mean[2]/std[2], (1.-mean[2])/std[2])
        return torch.cat((x_0, x_1, x_2), 0)
    while k_i == label and loop_i < max_iter:

        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)

        pert = torch.Tensor([np.inf])[0].to(device)
        f_kkk = torch.Tensor([np.inf])[0].to(device)
        w = torch.zeros(input_shape).to(device)

        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = copy.deepcopy(x.grad.data)
        var_min = 1000.
        for k in range(1, num_classes):
            print ("class:", k)
            zero_gradients(x)

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = copy.deepcopy(x.grad.data)

            w_k = cur_grad - grad_orig
            w_kk = smoothing(w_k)
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data

            pert_k = torch.abs(f_k) / w_kk.norm()
            r_i = torch.clamp(pert_k, min=1e-4) * w_kk*0.01 / w_kk.norm()

            test_img = copy.deepcopy(pert_image)

            result_img = test_img + r_i
            t_max_r = (1.-mean[0])/std[0] # 2.248
            t_max_g = (1.-mean[1])/std[1] # 2.428
            t_max_b = (1.-mean[2])/std[2] # 2.640

            threshold = torch.zeros([1, 3, 224, 224]).cuda()
            threshold[0, 0, :, :] = t_max_r
            threshold[0, 1, :, :] = t_max_g
            threshold[0, 2, :, :] = t_max_b

            diff = (result_img > threshold).type(torch.float32)
            n_overshoot = 224*224*3 - torch.sum(diff)
            print (torch.sum(diff))

        exit()






        pert_image = pert_image + r_i
        r_tot = r_tot + r_i


        # overshoot = result_img
        #
        # # find the ratio that prevents from sta
        #
        #
        #
        # test_img = copy.deepcopy(pert_image)
        # result_img = test_img + r_i
        # result_img_r = result_img[0, 0, :, :].view(-1)
        # result_img_g = result_img[0, 1, :, :].view(-1)
        # result_img_b = result_img[0, 2, :, :].view(-1)
        #
        # source_img_r = pert_image[0, 0, :, :].view(-1)
        # source_img_g = pert_image[0, 1, :, :].view(-1)
        # r_i_r = r_i[0, 0, :, :].view(-1)
        # r_i_g = r_i[0, 1, :, :].view(-1)
        #
        # t_max_r = (1.-mean[0])/std[0] # 2.248
        # t_max_g = (1.-mean[1])/std[1] # 2.428
        # t_max_b = (1.-mean[2])/std[2] # 2.640
        #
        # # print t_max_r, t_max_g, t_max_b
        #
        # overshoot_r = torch.max(result_img_r - t_max_r).item()
        # overshoot_g = torch.max(result_img_g - t_max_g).item()
        # overshoot_b = torch.max(result_img_b - t_max_b).item()
        # overshoot_r = (np.maximum(overshoot_r, 0.)==0.).astype(np.float32)
        # overshoot_g = (np.maximum(overshoot_g, 0.)==0.).astype(np.float32)
        # overshoot_b = (np.maximum(overshoot_b, 0.)==0.).astype(np.float32)
        # print overshoot_r, overshoot_b, overshoot_g
        # # print overshoot_r
        # # print overshoot_b
        # # print overshoot_g
        # # exit()
        # #
        # #
        # # print r_i.size()
        # # print pert_image.size()
        #
        # pert_image[0, 0, :, :] = pert_image[0, 0, :, :] + overshoot_r*r_i[0, 0, :, :]
        # pert_image[0, 1, :, :] = pert_image[0, 1, :, :] + overshoot_g*r_i[0, 1, :, :]
        # pert_image[0, 2, :, :] = pert_image[0, 2, :, :] + overshoot_b*r_i[0, 2, :, :]
        #
        #
        # r_tot[0, 0, :, :] = r_tot[0, 0, :, :] + overshoot_r*r_i[0, 0, :, :]
        # r_tot[0, 1, :, :] = r_tot[0, 1, :, :] + overshoot_g*r_i[0, 1, :, :]
        # r_tot[0, 2, :, :] = r_tot[0, 2, :, :] + overshoot_b*r_i[0, 2, :, :]

        # argmax_r = overshoot_r.argmax()
        # argmax_g = overshoot_g.argmax()
        # argmax_b = overshoot_b.argmax()
        # print source_img_g[argmax_g]
        # print r_i_g[argmax_g]
        # print overshoot_g.max(), "overshoot"
        # exit()



        # r_tot = r_tot + r_i


        # r_tot = torch.clamp(r_tot, -0.5, 0.5)

        # pert_image = image + r_tot










        # pert_image = pert_image + r_i
        # pert_image = clip_value(pert_image)





        check_fool = image + (1. + overshoot) * r_tot
        # check_fool = clip_value(check_fool)
        k_i = torch.argmax(net.forward(Variable(check_fool, requires_grad=True)).data).item()
        print (torch.max(net.forward(Variable(check_fool, requires_grad=True)).data).item())

        loop_i += 1

    x = Variable(pert_image, requires_grad=True)
    fs = net.forward(x)
    (fs[0, k_i] - fs[0, label]).backward(retain_graph=True)
    grad = copy.deepcopy(x.grad.data)
    grad = grad / grad.norm()

    fshow = r_tot.data.cpu().numpy().squeeze()
    print (fshow.var(), "variance")
    print (torch.norm(r_tot.view(-1), 2), "norm2")
    fshow = normalize(fshow)
    fshow = np.transpose(fshow, [1, 2, 0])
    # plt.imshow(fshow)
    # plt.show()

    # r_tot = lambda_fac * r_tot
    # pert_image = image + r_tot
    # pert_image = clip_value(pert_image)



    return grad, pert_image, r_tot, label, k_i, loop_i




def deepfool_var(im, net, lambda_fac=1., num_classes=50, overshoot=0.02, max_iter=20, device='cuda'):
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
        var_min = 1000.
        for k in range(1, num_classes):
            zero_gradients(x)

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = copy.deepcopy(x.grad.data)

            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data

            pert_k = torch.abs(f_k) / w_k.norm()


            var = w_k.var()

            if var < var_min:
            # if pert_k < pert:
                var_min = var
                pert = pert_k + 0.
                w = w_k + 0.

        r_i = torch.clamp(pert, min=1e-4) * w / w.norm()
        r_tot = r_tot + r_i

        pert_image = pert_image + r_i

        check_fool = image + (1. + overshoot) * r_tot
        k_i = torch.argmax(net.forward(Variable(check_fool, requires_grad=True)).data).item()

        loop_i += 1

    if loop_i==max_iter:
        print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    x = Variable(pert_image, requires_grad=True)
    fs = net.forward(x)
    (fs[0, k_i] - fs[0, label]).backward(retain_graph=True)
    grad = copy.deepcopy(x.grad.data)
    grad = grad / grad.norm()

    r_tot = lambda_fac * r_tot
    pert_image = image + r_tot

    return grad, pert_image

def deepfool_var2(im, net, lambda_fac, seg, num_classes=50, overshoot=0.02, max_iter=20, device='cuda'):
    print (lambda_fac, "#########################")
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
    n_segs = np.max(seg)+1
    seg_t = torch.from_numpy(seg).cuda().view(1, 1, 224, 224)
    while k_i == label and loop_i < max_iter:

        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)

        pert = torch.Tensor([np.inf])[0].to(device)
        w = torch.zeros(input_shape).to(device)

        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = copy.deepcopy(x.grad.data)
        var_min = 1000.
        for k in range(1, num_classes):
            zero_gradients(x)

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = copy.deepcopy(x.grad.data)

            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data

            pert_k = torch.abs(f_k) / w_k.norm()

            var = 0.
            for df in range(n_segs):
                w_seg = w_k.data.cpu().numpy() * (seg==df)
                var = var + np.sign(w_seg).var()


            if var < var_min:
            # if pert_k < pert:
                var_min = var
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

    return grad, pert_image




