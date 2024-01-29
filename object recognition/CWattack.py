from turtle import forward
import torchvision.models as models
import cv2
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torch
import warnings
class CWattack():
    def __init__(self,lr=0.01,step=2000,c=2,k=100) -> None:
        self.lr=lr
        self.step=step
        self.c=c#权衡两个损失函数的比重
        self.k=k#调整对抗样本的置信度
        
    def forward(self,net,im0,im1,target=False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        net.to(device)
        im0=torch.unsqueeze(im0,0).to(device)
        im1=torch.unsqueeze(im1,0).to(device)
        out0=net(im0).to(device)
        out1=net(im1).to(device)
        
        y0=np.array(out0[0].data.cpu()).argsort()[::-1]
        id_max=y0[0]#原始图像最大的输出类别
        
        y1=np.array(out1[0].data.cpu()).argsort()[::-1]
        id_target=y1[0]#有目标攻击的类别
        
        def f(x,k):#构造损失函数使得目标的类别的输出概率提高
            out=net(x).to(device)
            y=np.array(out[0].data.cpu()).argsort()[::-1]
            id_max_in=y[0]
            id_second=y[1]
            label_max=out[0,id_max_in]
            label_second=out[0,id_second]
            label_target=out[0,id_target]
            one_hot_label=torch.eye(len(out[0]))[id_target].to(device)
            except_target,_=torch.max((1-one_hot_label)*out,dim=1)
            if target:
                return torch.clamp(except_target-label_target,min=-k)#有目标攻击
            else:
                return torch.clamp(label_max-label_second,min=-k)#无目标攻击
            
        w=torch.zeros_like(im0).to(device)
        w.requires_grad=True
        optimizer=torch.optim.Adam([w],lr=self.lr)
        prev=1e10
        loop_i = 0
        k_i=id_max
        k=self.k
        c=self.c
        attack_image=im0
        for i in range(self.step):
            a=1/2*(torch.nn.Tanh()(w)+1)#利用tanh把图片的大小固定到（0，1）
            loss1=torch.nn.MSELoss(reduction='sum')(im0,a)#利用mseloss计算干扰图和原图的差距
            loss2=self.c*f(a,self.k)            
            loss=loss1+loss2
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            if loop_i % 100==0:
                if loss>prev:
                    warnings.warn('the attack can not converge')#如果发现多次迭代后梯度没有下降，发出警告终止迭代
                    attack_image=1/2*(torch.nn.Tanh()(w)+1)
                prev=loss    
            attack_image=1/2*(torch.nn.Tanh()(w)+1)
            fs = net.forward(attack_image)
            if loss1>=100:
                self.c=0
            elif loss1>=50:
                self.c=c
                self.k=k
            else:
                self.c=c/2
                self.k=k/10
            loop_i +=1
            
        im0_label=id_max
        im1_label=id_target
        out_attack=net(attack_image).to(device)
        y_attack=np.array(out_attack[0].data.cpu()).argsort()[::-1]
        id_attack=y_attack[0]
        attack_lable=id_attack
        r_tot=(attack_image-im0).cpu().detach().numpy()
        return loss1,loss2,r_tot,loop_i,im0_label,im1_label,attack_lable,attack_image