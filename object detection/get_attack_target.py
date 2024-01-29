import numpy as np
import torch

def get_mask(adv_tensor, mask, increment, attack_pixel_num, cal_num, batch_idx, count, idal_pixel_num):
    grad = adv_tensor.grad.detach().numpy()
    temp = 1 - mask  # Exclude already selected points
    remain_grad = np.multiply(grad, temp)
    get_grad = np.abs(remain_grad)
    get_total_grad = get_grad[0, 0, :, :] + get_grad[0, 1, :, :] + get_grad[0, 2, :, :]
    grad_reshape = get_total_grad.reshape(1, 416 * 416)
    # Sort the remaining points
    find_max = grad_reshape[0].argsort()[-(attack_pixel_num - cal_num):][::-1]
    # Randomly select 'increment' number of points
    rand_x = np.random.choice(find_max, increment)
    row = np.floor(rand_x / 416)
    col = (rand_x - row * 416)
    length = len(col)
    row = row.astype(np.int32)  # Get the rows and columns of the attack points
    col = col.astype(np.int32)
    Mask = np.zeros([1, 3, 416, 416])
    # Construct a new mask
    for i in range(length):
        Mask[0][0][row[i]][col[i]] = 1
        Mask[0][1][row[i]][col[i]] = 1
        Mask[0][2][row[i]][col[i]] = 1
    mask += Mask
    Mask_tensor = torch.from_numpy(mask).float().to('cpu')
    return mask, Mask_tensor
