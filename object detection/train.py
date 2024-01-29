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

parser = argparse.ArgumentParser()
# Number of subprocesses used for data loading, 0 means no subprocess, 1 means a total of two processes, and so on
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
# Number of epochs
parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train for')
# Whether to use GPU for training
parser.add_argument('--cuda', action='store_true', help='enables cuda')
# The target class's index number in ImageNet
parser.add_argument('--target', type=int, default=0, help='The target class: 859 == toaster')
# The target for the attack, used to turn one class into another
parser.add_argument('--totarget', type=int, default=16, help='The target class: 16 == dog')
# Maximum number of iterations to train the patch
parser.add_argument('--max_count', type=int, default=200, help='max number of iterations to find adversarial example')
# The shape category of the patch
parser.add_argument('--patch_type', type=str, default='circle', help='patch type: circle or square')
# Percentage of the attack
parser.add_argument('--percent', type=float, default=0.05, help='size 2% of 416x416 image')
# Number of pixels attacked each round
parser.add_argument('--increment', type=int, default=50, help='nums of attacked pixels')
# Number of training set samples
parser.add_argument('--train_size', type=int, default=100, help='Number of training images')
# Number of test set samples
parser.add_argument('--test_size', type=int, default=0, help='Number of test images')
# Image size combined with patch size to determine the final pixel value of the patch
parser.add_argument('--image_size', type=int, default=416, help='the height / width of the input image to network')
# Plot_all indicates whether to add patch images in the folder
parser.add_argument('--plot_all', type=int, default=1, help='1 == plot all successful adversarial images')
# Indicates which attack target model is
parser.add_argument('--netClassifier', default='yolov3', help="The target classifier")
# Output the training results to the target directory
parser.add_argument('--outf', default='./logs', help='folder to output images and model checkpoints')
parser.add_argument('--advspec', default='./advspec', help='folder to adv_specimen')
parser.add_argument('--patch', default='./patch', help='folder to patch')
parser.add_argument('--pth', default='./pth', help='folder to tensor_pth')
parser.add_argument('--loss_hot', default='./loss_hot', help='folder to loss_hot')
parser.add_argument('--loss_curve', default='./loss_curve', help='folder to loss_curve')
parser.add_argument('--prcurve', default='./prcurve', help='folder to prcurve')
# Set the random seed
parser.add_argument('--manualSeed', type=int, default=11, help='manual seed')

opt = parser.parse_args()
json_path = "./new.json"
img_path = "./images"  # Folder where images are stored
coco = COCO(annotation_file=json_path)
ids = list(sorted(coco.imgs.keys()))
# Construct directory
try:
    os.makedirs(opt.outf)
except OSError:
    pass
# Random seed
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
# Load the seed
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
cudnn.benchmark = True  # Improve runtime efficiency
# Load parameters
# Define some other parameters
iou_threshold = 0.45  # IoU threshold
target = opt.target
conf_target = opt.conf_target
max_count = opt.max_count
patch_type = opt.patch_type
patch_size = opt.patch_size
image_size = opt.image_size
train_size = opt.train_size
test_size = opt.test_size
plot_all = opt.plot_all

# The training set plus the test set should not exceed the total number of samples
assert train_size + test_size <= 100, "Training set size + Test set size > Total dataset size"
# Load model, netClassifier is the classifier being attacked here
netClassifier = pretrainedmodels.__dict__[opt.netClassifier](num_classes=80, pretrained=’coco’)
# Load COCO dataset
coco = COCO('D:/Dataset/coco/stuff_val2017.json')

# DataLoader setup
train_loader = torch.utils.data.DataLoader(
    dset.CocoDetection('D:/pattern_reconginition/adversarial-patch-master/adversarial-patch-master/images',
                       'D:/pattern_reconginition/adversarial-patch-master/adversarial-patch-master/new.json', 
                       transform=transforms.Compose([
                           transforms.ToTensor(),  # Convert image to tensor (range becomes 0~1)
                           ToSpaceBGR(netClassifier.input_space == 'BGR')  # Convert to RGB if original classifier input space is not RGB
                       ])),
    batch_size=1, 
    shuffle=False,  # Only one sample can be input for training at a time
    num_workers=opt.workers, 
    pin_memory=True, 
    drop_last=True
)

def train():
    print('=====> Begin TRAINING..............................................................')
    netClassifier.eval()
    # Counter, recording or setting a standard
    total = 0
    input_size = 416
    iou_threshold = 0.45
    score_threshold = 0.3
    results = []  # Store prediction results
    # Load images from train_loader
    for batch_idx, (data, labels) in enumerate(train_loader):
        img_idx = ids[batch_idx]
        print('idx: ', img_idx)
        print('==> TRAINING...........................................')
        # Convert data to a tensor with added grayscale bar
        input_tensor, data_numpy = tran_tensor_to_numpy(data)
        input_tensor = Variable(input_tensor.data, requires_grad=True)

        # Get some data of the original image
        T, C, W, H = data.shape
        ori_tensor, ori_numpy = tran_tensor_to_tensor(data, input_tensor, [W, H])
        # Save the initial sample
        cv2.imwrite("./%s/%d_original.png" % (opt.advspec, batch_idx + 1), ori_numpy)
        device = torch.device("cpu")
        netClassifier.to(device)
        # Original prediction results
        out_l, out_m, out_s = netClassifier(input_tensor)
        # Equivalent to out_1, out_2, out_3 = netClassifier.forward(input_tensor), similar to __call__ method
        # 13*13*3 + 26*26*3 + 52*52*3 = 10647, including 3 predicted priors centered on the sampling block, totaling 10647 priors
        out_l = out_l.to("cpu")  # l is [1, 255, 13, 13], 13x13 detection box 255 = 3x(80+5)
        out_m = out_m.to("cpu")  # m is [1, 255, 26, 26], 26x26 detection box 255 = 3x(80+5)
        out_s = out_s.to("cpu")  # s is [1, 255, 52, 52], 52x52 detection box 255 = 3x(80+5)
        # decode to get 85-dimensional vector information in the corresponding box, out_pred is [1, 10647, 85] 1 x [(13x13 + 26x26 + 52x52) x 3] x 85
        # Includes 80+5 dimensional vectors, 80 (probability of 80 types of objects in the box) + 5 (center x,y, width w, height h, and probability of containing objects)
        out_pred = netClassifier.predict(out_l, out_m, out_s)  # Predict all boxes based on YOLO output
        # One image outputs 10647 prediction boxes
        # 10647 = (52*52 + 26*26 + 13*13) * 3
        # Each box is an 85-dimensional vector, including:
        # The center x, y, width w, height h, probability p of the prediction box, followed by probabilities of various categories

        # Construct loss
        loss1, saveLoss, list = construct_loss(out_pred, out_pred, list, 1)
        # Update backwards only when loss is above the minimum limit, otherwise, it may cause errors
        if saveLoss.data > 0.001:
            # Backward pass
            saveLoss.backward()
            remain_grad = input_tensor.grad

            if remain_grad is not None:
                remain_grad = remain_grad.detach()
                get_grad = np.abs(remain_grad).numpy()
                get_total_grad = get_grad[0, 0, :, :] + get_grad[0, 1, :, :] + get_grad[0, 2, :, :]
                # The following saves the heatmap
                vmax = np.max(get_total_grad)
                vmin = np.min(get_total_grad)
                plt.imshow(get_total_grad, cmap=plt.cm.hot, vmin=vmin, vmax=vmax)
                plt.colorbar()
                plt.savefig(r"./%s/%d_original_loss.png" % (opt.loss_hot, batch_idx + 1))
                plt.clf()
            # Define the object for backpropagation
    input_tensor = Variable(input_tensor.data, requires_grad=False)
    # The following is the process of making boxes and non-maximum suppression to produce recognition and detection results before the attack
    bboxes, coors, scores, classes = postprocess_boxes(out_pred, data_numpy, input_size=input_size,
                                                    score_threshold=score_threshold)

    # Check if the attack target exists, skip the image if it does not exist
    key = np.in1d([opt.target], classes)
    if key == True:
        # Enter the attack module, get the results after the attack and store them in results
        results = attack2(data, percent, out_pred, input_size, score_threshold, batch_idx,
                        img_idx, results, ideal_percent)

    # The results include the X, Y, W, H of the attack target and the class score
    json_str = json.dumps(results, indent=4)  # Save the results to a JSON file
    with open('predict_results_without_attack.json', 'w') as json_file:
        json_file.write(json_str)

    # The following is the evaluation part
    coco_true = COCO(annotation_file="new.json")  # Load the actual results
    # Load the results after the attack
    coco_pre = coco_true.loadRes('predict_results_without_attack.json')  # Load the prediction results

    # Evaluation
    coco_evaluator = COCOeval(cocoGt=coco_true, cocoDt=coco_pre, iouType="bbox")
    coco_evaluator.evaluate()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    # The following is to extract data from the COCO toolkit and draw the PR curve
    pr_array1 = coco_evaluator.eval['precision'][0, :, 0, 0, 2]
    pr_array2 = coco_evaluator.eval['precision'][0, :, 0, 1, 2]
    pr_array3 = coco_evaluator.eval['precision'][0, :, 0, 2, 2]
    pr_array4 = coco_evaluator.eval['precision'][0, :, 0, 3, 2]
    x = np.arange(0.0, 1.01, 0.01)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.01)
    plt.grid(True)
    plt.plot(x, pr_array1, 'b-', label='all')
    plt.plot(x, pr_array2, 'c-', label='small')
    plt.plot(x, pr_array3, 'y-', label='medium')
    plt.plot(x, pr_array4, 'r-', label='large')
    # plt.xticks(x_1, x_1)
    plt.title("iou=0.5 catid=person maxdet=100")
    plt.legend(loc="lower left")
    plt.show()



if __name__ == '__main__':
    train()
  
    
