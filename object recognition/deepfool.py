import numpy as np
from torch.autograd import Variable
import torch as torch
import copy



def deepfool(image0, image1, net, overshoot=0.02, max_iter=100):

    """
       :param image0: initial image
       :param image1: target image
       :param net: network (resnet34)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool 
       :return: minimal perturbation , number of iterations, initial label, target label and perturbed image0
    """
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        print("Using GPU")
        image0 = image0.cuda()
        image1 = image1.cuda()
        net = net.cuda()
    else:
        print("Using CPU")

    f_image0 = net.forward(torch.tensor(image0[None,:,:,:], requires_grad=True)).data.cpu().numpy().flatten()
    f_image1 = net.forward(torch.tensor(image1[None,:,:,:], requires_grad=True)).data.cpu().numpy().flatten()
    
    #分别找出原始图片和目标图片的最大输出
    I0 = (np.array(f_image0)).flatten().argsort()[::-1]
    I1 = (np.array(f_image1)).flatten().argsort()[::-1]
    I0_max = I0[0]
    I1_max = I1[0]
    
    input_shape = image0.cpu().numpy().shape
    pert_image0 = copy.deepcopy(image0)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)#干扰总和

    loop_i = 0

    x = Variable(pert_image0[None, :], requires_grad=True)
    fs = net.forward(x)
    k_i = I0_max

    while k_i != I1_max and loop_i < max_iter:#判断是否干扰成功，或者迭代次数达到上限

        fs[0, I0_max].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        x.grad.zero_()
        fs[0, I1_max].backward(retain_graph=True)
        cur_grad = x.grad.data.cpu().numpy().copy()

        w = cur_grad - grad_orig#梯度差
        f = (fs[0, I1_max] - fs[0, I0_max]).data.cpu().numpy()#函数值差
        pert = abs(f)/np.linalg.norm(w.flatten())
        r_i =  (pert+1e-4) * w / np.linalg.norm(w)#单次干扰
        r_tot = np.float32(r_tot + r_i)

        if is_cuda:
            pert_image0 = image0 + (1+overshoot)*torch.from_numpy(r_tot).cuda()
        else:
            pert_image0 = image0 + (1+overshoot)*torch.from_numpy(r_tot)

        x = Variable(pert_image0, requires_grad=True)
        fs = net.forward(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())

        loop_i += 1

    r_tot = (1+overshoot)*r_tot

    return r_tot, loop_i, I0_max,I1_max, k_i, pert_image0