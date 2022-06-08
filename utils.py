import numpy as np
import torch


"""
Intersection over Union

N samples pred size: N x C x H x W

Returns (list of intersections per class, list of unions per class)

"""

# calculate the intersection over union
def iou(pred, gt_masks):

    # print('pred shape', pred.shape)
    ious = []
    num_class = len(gt_masks[0])

    unions = []
    intersections = []

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        one_hot_pred = one_hot_encode(pred).cuda()
    else:
        one_hot_pred = one_hot_encode(pred)

#     for cls in range(n_class - 1): # Leave out Unlabeled class

    # for each class, get a binary matrix
    for cls in range(num_class): # no "unlabeled" class (there are 4 channels and 4 labels)
        if use_gpu:
            c = torch.tensor([cls]).cuda()
        else:
            c = torch.tensor([cls])

        # https://pytorch.org/docs/stable/generated/torch.index_select.html

        # get the cth channel of the 
        p = torch.index_select(one_hot_pred, 1, c)
        t = torch.index_select(gt_masks, 1, c)

        # Should be of shape N x 1 x H x W
        # print('p shape:', p.shape)
        # print('t shape:', t.shape)

        # find intersection and union for each of the N pred/gt maps
        intersection = torch.sum(torch.logical_and(p, t)) # intersection calculation
        union = torch.sum(torch.logical_or(p, t)) # Union calculation
#         if union == 0:
#             ious.append(float('nan')) # if there is no ground truth, do not include in evaluation
#         else:
        # Append the calculated IoU to the list ious
        unions.append(union)
        intersections.append(intersection)
    return torch.tensor(intersections), torch.tensor(unions)


# calculates the number of matching valued pixels between pred (batchsize x H x W of ints) and gt (batchsize x H x W of ints)
def pixel_acc(pred, gt):

    num_samples, height, width = gt.shape
    total_num_pixels = num_samples * height * width
    
    num_correct = 0
    for n in range(num_samples):
        # Don't include background in pixel accuracy
        num_correct += torch.count_nonzero(torch.eq(pred[n], gt[n]))
    
    return num_correct / total_num_pixels

"""
Predictions of size N x C x H x W
"""

# take in a prediction (batchsize x num_classes x H x W)
def one_hot_encode(pred):
    # One hot encode the predictions
    num_samples, num_class, height, width = pred.shape

    # for each num_classes x H x W sample, return 
    for n in range(num_samples):
        # pred[n] shape: C x H x W
        indices = torch.argmax(pred[n], dim=0) # H x W

        # initialize an empty tensor of dim num_classes x H x W sample
        one_hot_pred = torch.zeros(num_class, height, width)

        # fill one_hot_pred with 1s at location of argmax across num_classes
        for c in range(num_class):
            one_hot_pred[c][indices == c] = 1

        # replace non-argmaxed sample with argmaxed sample
        pred[n] = one_hot_pred

    return pred