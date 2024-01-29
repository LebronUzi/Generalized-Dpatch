import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import argparse
import numpy as np
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import os
import random
from train import opt
from loss_set import construct_loss

# The following is the main attack module
'''
 input:
 x - original image data;
 percent - percentage of the image to be attacked
 out_pred - prediction results of the original image
 input_size - input data size
 score_threshold - discrimination threshold
 batch_idx - current image number
 img_id - current image ID number
 results - to store the final result data
 ideal_percent - ideal percentage used only for testing purposes
'''
def attack2(x, percent, out_pred, input_size, score_threshold, batch_idx, img_id, results,
            ideal_percent):
    mask = np.zeros([1, 3, 416, 416])  # mask for determining attack points
    cal_num = 0  # record the cumulative number of attacks
    increment = opt.increment  # the number of attack pixels added each round
    list = []  # for record keeping
    # Some variables for defining, saving, or converting data
    remember_best_tensor = torch.zeros(1, 3, 416, 416)
    remember_loss = 65535 * torch.ones(1)
    remember_grad = torch.zeros(1, 3, 416, 416)
    T, C, W, H = x.shape  # Shape of the original data, used for transformation
    cumulative_grad = torch.zeros([1, 3, 416, 416])  # cumulative gradient
    Mask_tensor = torch.zeros([1, 3, 416, 416])  # tensor corresponding to the mask
    attack_pixel_num = int(percent * 416 * 416)  # determine the total number of attack points
    print("Total attack pixels:", attack_pixel_num, "Iterations per round:", increment)
    ideal_pixel_num = int(ideal_percent * 416 * 416)  # ideal number of attack points
    LOSS = []  # record results
    COUNT = []
    print("attack2 begin")
    persistent_num = 0  # record the number of rounds successfully disrupted by the attack
    opt.netClassifier.eval()
    adv_x = x  # define the attacked image
    count = 0  # counter to exit after exceeding the limit
    adv_tensor, adv_numpy = tran_tensor_to_numpy(adv_x)  # add grayscale bar suitable for yolo input
    # Save the original tensor for cumulative gradient updates
    original_tensor = adv_tensor.data
    original_tensor.requires_grad = False

    adv_tensor = Variable(adv_tensor.data, requires_grad=True)
    # Define Adam optimizer
    Optimizer = torch.optim.Adam([adv_tensor], lr=0.01, amsgrad=True)
    key = True  # Key can be replaced with other requirements; it's designed this way here
    while (key == True) and (persistent_num < 4):
        count += 1
        COUNT.append(count)  # for plotting
        # Network output results after adding attack pixels
        adv_l, adv_m, adv_s = opt.netClassifier(adv_tensor)
        adv_pred = opt.netClassifier.predict(adv_l, adv_m, adv_s)

        # Construct loss, loss2 was designed for initial comparison and can be removed
        loss, loss2, list = construct_loss(adv_pred, out_pred, list, count)
        # Record changes in loss
        LOSS.append(loss.data)
        # Save the tensor and corresponding gradient of the smallest loss
        if (loss.data < remember_loss.data):
            remember_best_tensor.data = adv_tensor.data
            remember_loss = loss.data
            if (loss.data > 0.001):
                remember_grad = adv_tensor.grad
            Optimizer.zero_grad()  # Clear gradients
            if (loss.data / len(list) < 0.1):  # Standard for loss to start attack detection and check if it meets our requirements
                trial_bboxes, trial_coors, trial_scores, trial_classes = postprocess_boxes(adv_pred,
                                                                                           adv_numpy, input_size=input_size, score_threshold=score_threshold)
                # Replace with another_key, which is our defined standard for achieving the expected results
                another_key = np.in1d([opt.target], trial_classes)
                # another_key = len(trial_classes)
                print('Is the attack successful:', another_key)
                if another_key == False:
                    persistent_num += 1
                    print("Correct attack, entering the stability judgment stage, persist_num = ", persistent_num)
                else:
                    print("Loss is already below a certain value, judging if the attack is successful!")
            elif (loss.data / len(list) > 0.1 and persistent_num > 0):  # Exit after reaching the standard
                persistent_num -= 1
            # Backward derivative and cumulative gradient update when loss is above the lower limit
            if loss.data > 0.0001:
                loss.backward()
                if (cal_num < attack_pixel_num):
                    # Get the mask, which is the number of attack points
                    mask, Mask_tensor = get_mask(adv_tensor, mask, increment, attack_pixel_num, cal_num,
                                                 batch_idx, count, ideal_pixel_num)
                    cal_num += increment
                    renew_grad = torch.mul(Mask_tensor, adv_tensor.grad).detach()
                    # Limit the range of gradient updates per iteration: to |0.005| to achieve a more precise attack
                    renew_grad.data.clamp_(-0.005, 0.005)  # Control the limit of each update
                    cumulative_grad += renew_grad
                    cumulative_grad.data.clamp_(-0.06, 0.06)  # Control the cumulative update limit
                    adv_tensor.data = original_tensor.data - cumulative_grad.data  # Update the sample
                    adv_tensor.data.clamp_(0, 1)  # Control the pixel range to 0~255
                else:
                    print("Loss is small enough! Attack successful!")
                    break