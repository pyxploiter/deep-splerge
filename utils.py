import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage import transform as sktsf
from torchvision import transforms as tvtsf

def get_logits(sig_probs):
    """
    Arguments:
    ----------
    sig_probs: output sigmoid probs from model
    """

    pos = sig_probs.squeeze(dim=0).view(sig_probs.shape[0],sig_probs.shape[2],1)
    neg = torch.sub(1, sig_probs.squeeze(dim=0)).view(sig_probs.shape[0],sig_probs.shape[2],1)
    logits = torch.cat((pos,neg),2)
    return logits

def cross_entropy_loss(logits, targets):
    """
    Arguments:
    ----------
    logits: (N, num_classes)
    targets: (N)
    """
    # print(logits.shape, targets.shape)
    # print(torch.abs(x-y))
    log_prob = -1.0 * F.log_softmax(logits, 1)
    loss = log_prob.gather(2, targets.unsqueeze(2))
    loss = loss.mean()
    return loss

def split_loss(outputs, targets):
    """
    Arguments:
    ----------
    outputs: (rpn_outputs, cpn_outputs)
    targets: (rpn_targets, cpn_targets)
    """
    
    lambda3 = 0.1
    lambda4 = 0.25

    rpn_outputs, cpn_outputs = outputs
    rpn_targets, cpn_targets = targets

    r3, r4, r5 = rpn_outputs
    c3, c4, c5 = cpn_outputs
    

    # TODO: update cross entropy loss acc. to the paper
    crit = torch.nn.BCELoss()
    rpn_targets = rpn_targets.float()
    cpn_targets = cpn_targets.float()
    
    # print(rpn_targets.shape, r3.shape, r3.squeeze(1).shape)
    rl3 = crit(r3.squeeze(1), rpn_targets)
    rl4 = crit(r4.squeeze(1), rpn_targets)
    rl5 = crit(r5.squeeze(1), rpn_targets)

    cl3 = crit(c3.squeeze(1), cpn_targets)
    cl4 = crit(c4.squeeze(1), cpn_targets)
    cl5 = crit(c5.squeeze(1), cpn_targets)

    rpn_loss = rl5 + (lambda4 * rl4) + (lambda3 * rl3)
    cpn_loss = cl5 + (lambda4 * cl4) + (lambda3 * cl3)
    
    loss = rpn_loss + cpn_loss

    return loss, rpn_loss, cpn_loss

def collate_fn(batch):
    return tuple(zip(*batch))

def getBack(var_grad_fn):
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:

                tensor = getattr(n[0], 'variable')
                print(n[0])
                # print('tensor with grad found:', tensor)
                print(' - gradient:', tensor.grad)
                print(' - grad_fn:', tensor.grad_fn)
                print()
            except AttributeError as e:
                print(e)
                getBack(n[0])


def normalize_numpy_image(image):
    # Normalizing image
    norm = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    image = norm(torch.from_numpy(image))

    return image

def resize_image(image, min_size=600, max_size=1024, fix_resize=False):
    # Rescaling Images
    C, H, W = image.shape
    min_size = min_size
    max_size = max_size
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    image = image / 255.

    if fix_resize:
        image = sktsf.resize(image, (C, min_size, max_size), mode='reflect',anti_aliasing=False)
    else:
        image = sktsf.resize(image, (C, H * scale, W * scale), mode='reflect',anti_aliasing=False)
        
    return image

def get_column_separators(image, smoothing=2, is_row=True):
    if is_row:
        cols = np.where(np.sum(image, axis=1)!=0)[0]
    else:
        cols = np.where(np.sum(image, axis=0)!=0)[0]

    if len(cols) == 0:
        return []

    adjacent_cols = [cols[0]]
    final_seperators = []
    for i in range(1, len(cols)):

        if cols[i] - cols[i - 1] < smoothing:
            adjacent_cols.append(cols[i])
            
        elif len(adjacent_cols) > 0:
            final_seperators.append(sum(adjacent_cols) // len(adjacent_cols))
            adjacent_cols = [cols[i]]

    if len(adjacent_cols) > 0:
        final_seperators.append(sum(adjacent_cols) // len(adjacent_cols))
    
    return final_seperators

def get_midpoints_from_grid(grid):

    row_sep = np.where(np.sum(grid, axis=1) == grid.shape[1])[0]
    col_sep = np.where(np.sum(grid, axis=0) == grid.shape[0])[0]

    def find_midpoint(indices):

        adj_indices = [indices[0]]
        midpoints = []

        for i in range(1, len(indices)):
            if indices[i] - indices[i - 1] == 1:
                adj_indices.append(indices[i])
            
            elif len(adj_indices) > 0:
                midpoints.append(sum(adj_indices) // len(adj_indices))
                adj_indices = [indices[i]]

        if len(adj_indices) > 0:
            midpoints.append(sum(adj_indices) // len(adj_indices))

        return midpoints

    col_midpoints, row_midpoints = [], []

    if len(row_sep):
        row_midpoints = find_midpoint(row_sep)

    if len(col_sep):
        col_midpoints = find_midpoint(col_sep)
        
    return row_midpoints, col_midpoints


def tensor_to_numpy_image(tensor, display=False, write_path=None, threshold=0.7):
    tensor = tensor.squeeze(0)  #1,c,h,w -> c,h,w
    c, h, w = tensor.shape
    np_array = np.array(tensor.view(h,w,c).detach())
    np_array[np_array > threshold] = 255
    np_array[np_array <= threshold] = 0

    if display:
        cv2.imshow("image"+str(torch.rand(3)), np_array)
        cv2.waitKey(0)
    if write_path:
        cv2.imwrite(write_path, np_array)

    return np_array

def probs_to_image(tensor, image_shape, axis):
    """this converts probabilities tensor to image"""
    # (1, 1, n) = tensor.shape
    # b,c,h,w = image_shape
    b, c, h, w = image_shape
    if axis == 0:
        tensor_image = tensor.view(1,1,tensor.shape[2]).repeat(1,h,1)
    
    elif axis == 1:
        tensor_image = tensor.view(1,tensor.shape[2],1).repeat(1,1,w)
    
    else:
        print("Error: invalid axis.")

    return tensor_image.unsqueeze(0) 

def binary_grid_from_prob_images(row_prob_img, col_prob_img, thresh=0.7, row_smooth=5, col_smooth=15):
    
    row_prob_img[row_prob_img > thresh] = 1
    row_prob_img[row_prob_img <= thresh] = 0
    
    col_prob_img[col_prob_img > thresh] = 1
    col_prob_img[col_prob_img <= thresh] = 0
        
    row_indices = get_column_separators(row_prob_img.squeeze(0).squeeze(0).detach().numpy(), smoothing=row_smooth, is_row=True)
    col_indices = get_column_separators(col_prob_img.squeeze(0).squeeze(0).detach().numpy(), smoothing=col_smooth, is_row=False)

    col_smooth_image = torch.zeros(col_prob_img.shape)
    row_smooth_image = torch.zeros(row_prob_img.shape)

    for i in col_indices:
        col_img = col_smooth_image[0][0].transpose(1,0)
        if (i > 0):
            col_img[i+1:i+4] = 1.0
            col_img[max(0,i-3):i+1] = 1.0

    row_img = row_smooth_image[0][0]
    for i in row_indices:
        if (i > 0):
            row_img[i+1:i+4] = 1.0
            row_img[max(0,i-3):i+1] = 1.0

    row_img = row_img.unsqueeze(0).unsqueeze(0) 

    grid = row_img.int() | col_smooth_image.int()
    grid = grid.float()
    
    return grid, row_img, col_smooth_image

