import cv2
import numpy as np
import torch
import torch.nn.functional as F

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

def splerge_loss(outputs, targets):
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
    
    # r3_logits = get_logits(r3)
    # r4_logits = get_logits(r4)
    # r5_logits = get_logits(r5)
    
    # c3_logits = get_logits(c3)
    # c4_logits = get_logits(c4)
    # c5_logits = get_logits(c5)

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

    # rl3 = cross_entropy_loss(r3_logits, rpn_targets)
    # rl4 = cross_entropy_loss(r4_logits, rpn_targets)
    # rl5 = cross_entropy_loss(r5_logits, rpn_targets)

    # cl3 = cross_entropy_loss(c3_logits, cpn_targets)
    # cl4 = cross_entropy_loss(c4_logits, cpn_targets)
    # cl5 = cross_entropy_loss(c5_logits, cpn_targets)

    rpn_loss = rl5 + (lambda4 * rl4) + (lambda3 * rl3)
    cpn_loss = cl5 + (lambda4 * cl4) + (lambda3 * cl3)
    
    loss = rpn_loss + cpn_loss

    return loss, rpn_loss, cpn_loss

def collate_fn(batch):
    return tuple(zip(*batch))

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

# def get_midpoints_from_grid(grid):
#     row_sep = np.where(np.sum(grid, axis=1) == grid.shape[1])[0]
#     col_sep = np.where(np.sum(grid, axis=0) == grid.shape[0])[0]

#     def find_midpoint(indices):
#         adj_indices = [indices]
#         midpoints = []

#         for i in range(1, len(indices)):
#             if indices - indices[i - 1] == 1:
#                 adj_indices.append(indices[i])
            
#             elif len(adj_indices) > 0:
#                 midpoints.append(sum(adj_indices) // len(adj_indices))
#                 adj_indices = [indices[i]]

#         if len(adj_indices) > 0:
#             midpoints.append(sum(adj_indices) // len(adj_indices))

#         return midpoints

#     if len(col_sep):
#         find_midpoint(col_sep)

#     if len(row_sep):
#         find_midpoint(row_sep)
        


def tensor_to_numpy_image(tensor, display=False, write_path=None):
    tensor = tensor.squeeze(0)  #1,c,h,w -> c,h,w
    c, h, w = tensor.shape
    np_array = np.array(tensor.view(h,w,c).detach())
    np_array[np_array > 0.7] = 255
    np_array[np_array <= 0.7] = 0

    if display:
        cv2.imshow("image"+str(torch.rand(3)), np_array)
        # cv2.waitKey(0)
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

def binary_grid_from_prob_images(row_prob_img, col_prob_img, thresh=0.7):
    
    row_prob_img[row_prob_img > thresh] = 1
    row_prob_img[row_prob_img <= thresh] = 0
    
    col_prob_img[col_prob_img > thresh] = 1
    col_prob_img[col_prob_img <= thresh] = 0
    
    # row_indices = np.unique(np.where(row_prob_img.squeeze(0).squeeze(0) == 1)[0])
    # col_indices = np.unique(np.where(col_prob_img.squeeze(0).squeeze(0).transpose(1,0) == 1)[0])
    
    row_indices = get_column_separators(row_prob_img.squeeze(0).squeeze(0).detach().numpy(), smoothing=5, is_row=True)
    col_indices = get_column_separators(col_prob_img.squeeze(0).squeeze(0).detach().numpy(), smoothing=5, is_row=False)

    # tensor_to_numpy_image(row_prob_img, True)
    # tensor_to_numpy_image(col_prob_img, True)

    # print("row ind:",row_indices)
    # print("col ind:",col_indices)

    for i in col_indices:
        trans_col_prob_img = col_prob_img[0][0].transpose(1,0)
        if (i > 0):
            trans_col_prob_img[i+1:i+4] = 1.0
            trans_col_prob_img[max(0,i-3):i] = 1.0

    row_prob_img = row_prob_img[0][0]
    for i in row_indices:
        if (i > 0):
            row_prob_img[i+1:i+4] = 1.0
            row_prob_img[max(0,i-3):i] = 1.0

    row_prob_img = row_prob_img.unsqueeze(0).unsqueeze(0) 

    grid = row_prob_img.int() | col_prob_img.int()
    grid = grid.float()
    # tensor_to_numpy_image(row_prob_img, True)
    # tensor_to_numpy_image(col_prob_img, True)
    # tensor_to_numpy_image(grid, True)
    # cv2.waitKey(0)
    
    return grid

