import torch
from train import opt

def construct_loss(adv_pred, truth_pred, list, count):
    conf = adv_pred[0, :, 4]  # Confidence, class probability, etc., of the original image
    prob = adv_pred[0, :, 5:]
    loss1_1 = torch.zeros([1, 1], dtype=torch.float)
    loss1_2 = torch.zeros([1, 1], dtype=torch.float)
    objectness = torch.sigmoid(conf)
    # Normalize probabilities of all classes
    num = 0
    # Original loss heatmap
    if count == 1:
        for i in range(len(conf)):
            score = torch.mul(conf[i], prob[i][0])
            if objectness[i] > 0.53:
                loss1_1 += objectness[i]
            if score > 0.2:
                num += 1
                list.append(i)

    totarget = opt.totarget
    # Constructing loss in a stepwise manner as below, can be replaced with different loss constructions as per different requirements
    for j in range(len(list)):
        score = torch.mul(prob[list[j]][totarget], conf[list[j]])
        if score < 0.2:
            loss1_2 += 10 * (1 - prob[list[j]][totarget])
        elif score < 0.4:
            loss1_2 += 5 * (1 - prob[list[j]][totarget])
        elif score < 0.6:
            loss1_2 += 1 - prob[list[j]][totarget]
        else:
            loss1_2 += 0.1 * (1 - prob[list[j]][totarget])
    loss = loss1_2
    return loss, loss1_1, list
