from sklearn.metrics import accuracy_score
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
import math
import torchvision.models as models
from PIL import Image
from deepfool import deepfool
import os
import cv2
import time

net = models.resnet34(pretrained=True)
net.eval()
image_path='images'
total=0
true_perturb=0
success_perturb=0
perturb=[]
loop=[]
start_time=time.time()
for i in range(100):
    im0_name='{}'.format(i+1).zfill(3)+r'\0.png'
    im1_name='{}'.format(i+1).zfill(3)+r'\1.png'
    im0_orig = Image.open(os.path.join(image_path,im0_name))
    im1_orig = Image.open(os.path.join(image_path,im1_name))

    mean = [ 0.485, 0.456, 0.406 ]
    std = [ 0.229, 0.224, 0.225 ]


    im0 = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(250),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean,
                            std = std)])(im0_orig)
    im1 = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(250),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean,
                            std = std)])(im1_orig)

    r,loop_i, label_orig, label_target,label_pert, image_pert = deepfool(im0, im1, net)


    def clip_tensor(A, minv, maxv):
        A = torch.max(A, minv*torch.ones(A.shape).cuda())
        A = torch.min(A, maxv*torch.ones(A.shape).cuda())
        return A

    clip = lambda x: clip_tensor(x, 0, 255)

    tf = transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=[1/0.229,1/0.224,1/0.225]),
                            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
                            transforms.Lambda(clip),
                            transforms.ToPILImage(),
                            transforms.CenterCrop(250)])
    image_pert=torch.reshape(image_pert,[3,250,250])
    image_pert=tf(image_pert)

    img_PIL = np.array(image_pert)
    img_PIL_cv2 = cv2.cvtColor(img_PIL,cv2.COLOR_RGB2BGR)
    test_name='{}'.format(i+1).zfill(3)+r'\test.png'
    cv2.imwrite(os.path.join(image_path,test_name), img_PIL_cv2) 
    print("Original label = ", label_orig)
    print("target label = ", label_target)
    print("perturb laebel=",label_pert)
    total +=1
    if label_target==label_pert:
        success_perturb += 1
        
    if label_pert!=label_orig:
        true_perturb += 1
    r_L2=np.linalg.norm(r)
    perturb.append(r_L2)
    loop.append(loop_i)
    # plt.figure()
    # plt.imshow(image_pert)
    # plt.title(label_pert) 
    # plt.show()
end_time=time.time()
print('total_time:',end_time-start_time)
x=np.arange(1,101)
accuracy_true=100*true_perturb/total
accuracy_success=100*success_perturb/total
print('the accuracy of successful pert is{}%,the accuracy of targeted pert is{}% '.format(accuracy_true,accuracy_success))
plt.figure(1)
a=plt.subplot(121)
a.set_title('perturb_size')
plt.plot(x,perturb)
b=plt.subplot(122)
b.set_title('iteration_times')
plt.plot(x,loop)
plt.show()