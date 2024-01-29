import matplotlib.pyplot as plt 
import onnxoptimizer
import torchvision.models as models
import cv2
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torch
import os
import CWattack
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

    net = models.resnet34(pretrained=True)
    net.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    im0 = transforms.Compose([
        transforms.ToTensor()])(im0_orig)
    im1 = transforms.Compose([
        transforms.ToTensor()])(im1_orig)

    attack=CWattack.CWattack()
    r,loop_i,label_orig, label_target,label_pert,attack_image=attack.forward(net,im0,im1,target=True)
        

    tf = transforms.Compose([
                            transforms.ToPILImage()
                        ])

    attack_image=torch.reshape(attack_image,[3,250,250])
    image_pert=tf(attack_image)


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

end_time=time.time()
print('total_time:',end_time-start_time)
x=np.arange(1,101)
accuracy_true=100*true_perturb/total
accuracy_success=100*success_perturb/total
print('the accuracy of successful pert is{}%,the accuracy of targeted pert is{}% '.format(accuracy_true,accuracy_success))
x=np.arange(1,101)
accuracy=true_perturb/total
plt.figure(1)
a=plt.subplot(121)
a.set_title('perturb_size')
plt.plot(x,perturb)
b=plt.subplot(122)
b.set_title('iteration_times')
plt.plot(x,loop)
plt.show()

net = models.resnet34(pretrained=True)
net.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

im0_orig = Image.open(r'D:\Codes\Python\Project\CV\Deepfool\images\001\0.png')
im1_orig = Image.open(r'D:\Codes\Python\Project\CV\Deepfool\images\001\1.png')

mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]

im0 = transforms.Compose([
    transforms.ToTensor()])(im0_orig)
im1 = transforms.Compose([
    transforms.ToTensor()])(im1_orig)
start_time=time.time()
attack=CWattack.CWattack()
loss1,loss2,r,loop_i,label_orig, label_target,label_pert,attack_image=attack.forward(net,im0,im1,target=True)
end_time=time.time()
print(end_time-start_time)
print("origin:{}".format(label_orig))
print("target:{}".format(label_target))
print("perturb:{}".format(label_pert))
print("loss:",loss1,"\n",loss2)

tf = transforms.Compose([
                        transforms.ToPILImage()
                    ])

attack_image=torch.reshape(attack_image,[3,250,250])
attack_image=tf(attack_image)

plt.figure()
plt.imshow(attack_image)
plt.show()
