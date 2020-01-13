import torch
import torch.nn.functional as F

class SFCN(torch.nn.Module):

    def __init__(self):
        super(SFCN, self).__init__()

        self.conv1 = torch.nn.Conv2d(8, 18, kernel_size=7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(18, 18, kernel_size=7, stride=1, padding=3)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=(2,2), stride=(1,1), padding=1)
        
        self.conv4 = torch.nn.Conv2d(18, 18, kernel_size=5, stride=1, padding=2)
        self.conv5 = torch.nn.Conv2d(18, 32, kernel_size=5, stride=1, padding=2)
        self.pool6 = torch.nn.MaxPool2d(kernel_size=(2,2), stride=(1,1))

    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool3(x)
        
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool6(x)

        return(x)

class Block(torch.nn.Module):

    def __init__(self, block_num):
        super(Block, self).__init__()

        self.block_num = block_num
        self.block_inputs = [32, 55, 55]
        self.block_conv1x1_output = 36
        self.block_output = 55

        self.dil_conv2d_1 = torch.nn.Conv2d(self.block_inputs[self.block_num-1], 6, kernel_size=5, dilation=1, stride=1, padding=2)
        self.dil_conv2d_2 = torch.nn.Conv2d(self.block_inputs[self.block_num-1], 6, kernel_size=5, dilation=2, stride=1, padding=4)
        self.dil_conv2d_3 = torch.nn.Conv2d(self.block_inputs[self.block_num-1], 6, kernel_size=5, dilation=3, stride=1, padding=6)

        # 1x1 convolution for top branch
        self.conv_1x1_top = torch.nn.Conv2d(18, self.block_conv1x1_output, kernel_size=1) 

        # 1x1 convolution for bottom branch
        self.conv_1x1_bottom = torch.nn.Conv2d(18, 1, kernel_size=1) 

    def forward(self, x, grid):
     
        batch_size, channels, height, width = x.shape
        
        x1 = F.relu(self.dil_conv2d_1(x))
        x2 = F.relu(self.dil_conv2d_2(x))
        x3 = F.relu(self.dil_conv2d_3(x))

        out_feature = torch.cat((x1, x2, x3), 1)

        #################### TOP BRANCH #######################
        top_branch_x = self.conv_1x1_top(out_feature)
        
        row_mids, col_mids = [0], [0]
        r_mids, c_mids = utils.get_midpoints_from_grid(grid.numpy())
        row_mids.extend(r_mids)
        row_mids.append(grid.shape[0])
        col_mids.extend(c_mids)
        col_mids.append(grid.shape[1])

        for row_id in range(1, len(row_mids)):
            for col_id in range(1, len(col_mids)):
                grid_mean = torch.mean(top_branch_x[:, :, row_mids[row_id-1]:row_mids[row_id], col_mids[col_id-1]: col_mids[col_id]])
                top_branch_x[:, :, row_mids[row_id-1]:row_mids[row_id], col_mids[col_id-1]: col_mids[col_id]] = grid_mean
        
        ################# BOTTOM BRANCH #######################
        bottom_branch_x = self.conv_1x1_bottom(out_feature)

        temp = torch.tensor((), dtype=torch.float32)
        bottom_branch_grid_pools = temp.new_zeros((batch_size, 1, len(row_mids)-1, len(col_mids)-1), requires_grad=True)

        for row_id in range(1, len(row_mids)):
            for col_id in range(1, len(col_mids)):
                grid_mean = torch.mean(bottom_branch_x[:, :, row_mids[row_id-1]:row_mids[row_id], col_mids[col_id-1]: col_mids[col_id]])
                bottom_branch_x[:, :, row_mids[row_id-1]:row_mids[row_id], col_mids[col_id-1]: col_mids[col_id]] = grid_mean
                bottom_branch_grid_pools[:, 0, row_id-1, col_id-1] = torch.sigmoid(grid_mean)
                
        bottom_branch_sig_probs = torch.sigmoid(bottom_branch_x)

        out_x = torch.cat((top_branch_x, out_feature, bottom_branch_sig_probs), 1)

        if self.block_num > 1:
            return (out_x, bottom_branch_grid_pools)

        return (out_x, None)


class Merge(torch.nn.Module):

    def __init__(self):
        super(Merge, self).__init__()
        self.blocks = 3
        self.sfcn = SFCN()
        
        self.up_block_1 = Block(block_num=1)
        self.up_block_2 = Block(block_num=2)
        self.up_block_3 = Block(block_num=3)

        self.down_block_1 = Block(block_num=1)
        self.down_block_2 = Block(block_num=2)
        self.down_block_3 = Block(block_num=3)

        self.left_block_1 = Block(block_num=1)
        self.left_block_2 = Block(block_num=2)
        self.left_block_3 = Block(block_num=3)

        self.right_block_1 = Block(block_num=1)
        self.right_block_2 = Block(block_num=2)
        self.right_block_3 = Block(block_num=3)

        # self.up_branch = torch.nn.Sequential(
        #                     Block(block_num=1),
        #                     Block(block_num=2),
        #                     Block(block_num=3),
        #                 )

        # self.down_branch = torch.nn.Sequential(
        #                     Block(block_num=1),
        #                     Block(block_num=2),
        #                     Block(block_num=3),
        #                 )

        # self.left_branch = torch.nn.Sequential(
        #                     Block(block_num=1),
        #                     Block(block_num=2),
        #                     Block(block_num=3),
        #                 )

        # self.right_branch = torch.nn.Sequential(
        #                     Block(block_num=1),
        #                     Block(block_num=2),
        #                     Block(block_num=3),
        #                 )

    def forward(self, x):
        grid = x[0][7].clone().detach()

        x = self.sfcn(x)

        up, _ = self.up_block_1(x, grid)
        up, up_probs_1 = self.up_block_2(up, grid)
        up, up_probs_2 = self.up_block_3(up, grid)
        up_out = [up_probs_1, up_probs_2]

        down, _ = self.down_block_1(x, grid)
        down, down_probs_1 = self.down_block_2(down, grid)
        down, down_probs_2 = self.down_block_3(down, grid)
        down_out = [down_probs_1, down_probs_2]

        D1_probs = torch.FloatTensor([1/2*up_probs_1[:,:,i+1,j]*down_probs_1[:,:,i,j] + 1/4*(up_probs_1[:,:,i+1,j]+down_probs_1[:,:,i,j]) 
            for i in range(up_probs_1.shape[2]-1) for j in range(up_probs_1.shape[3])])
        D1_probs = D1_probs.view(x.shape[0], 1, up_probs_1.shape[2]-1, up_probs_1.shape[3])

        D2_probs = torch.FloatTensor([1/2*up_probs_2[:,:,i+1,j]*down_probs_2[:,:,i,j] + 1/4*(up_probs_2[:,:,i+1,j]+down_probs_2[:,:,i,j]) 
            for i in range(up_probs_2.shape[2]-1) for j in range(up_probs_2.shape[3])])
        D2_probs = D2_probs.view(x.shape[0], 1, up_probs_2.shape[2]-1, up_probs_2.shape[3])
        
        left, _ = self.left_block_1(x, grid)
        left, left_probs_1 = self.left_block_2(left, grid)
        left, left_probs_2 = self.left_block_3(left, grid)
        left_out = [left_probs_1, left_probs_2]

        right, _ = self.right_block_1(x, grid)
        right, right_probs_1 = self.right_block_2(right, grid)
        right, right_probs_2 = self.right_block_3(right, grid)
        right_out = [right_probs_1, right_probs_2]

        R1_probs = torch.FloatTensor([1/2*left_probs_1[:,:,i,j+1]*right_probs_1[:,:,i,j] + 1/4*(left_probs_1[:,:,i,j+1]+right_probs_1[:,:,i,j]) 
            for i in range(left_probs_1.shape[2]) for j in range(left_probs_1.shape[3]-1)])
        R1_probs = R1_probs.view(x.shape[0], 1, left_probs_1.shape[2], left_probs_1.shape[3]-1)

        R2_probs = torch.FloatTensor([1/2*left_probs_2[:,:,i,j+1]*right_probs_2[:,:,i,j] + 1/4*(left_probs_2[:,:,i,j+1]+right_probs_2[:,:,i,j]) 
            for i in range(left_probs_2.shape[2]) for j in range(left_probs_2.shape[3]-1)])
        R2_probs = R2_probs.view(x.shape[0], 1, left_probs_1.shape[2], left_probs_1.shape[3]-1)

        return D1_probs, D2_probs, R1_probs, R2_probs

def merge_loss(outputs, targets):
    D1_probs, D2_probs, R1_probs, R2_probs = outputs

    td = torch.randint(0,2, (D1_probs.shape[2], D1_probs.shape[3])).float().unsqueeze(0)
    tr = torch.randint(0,2, (R1_probs.shape[2], R1_probs.shape[3])).float().unsqueeze(0)
    targets = [td, tr]
    D_targets, R_targets = targets
    
    crit = torch.nn.BCELoss()
    dl2 = crit(D1_probs.squeeze(1), D_targets)
    dl3 = crit(D2_probs.squeeze(1), D_targets)
    
    rl2 = crit(R1_probs.squeeze(1), R_targets)
    rl3 = crit(R2_probs.squeeze(1), R_targets)

    lambda2 = 0.25

    down_loss = dl3 + (lambda2 * dl2)
    right_loss = rl3 + (lambda2 * rl2)

    loss = down_loss + right_loss

    return loss, down_loss, right_loss


from torch.utils.data import DataLoader
from transforms import get_transform
from dataloader import TableDataset
import os
import utils
from splerge import Splerge
import numpy as np

print("Loading dataset...")
dataset = TableDataset(os.getcwd(), "data/images", "data/labels", get_transform(train=True), False)
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
train_dataset = torch.utils.data.Subset(dataset, indices)
train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False, num_workers=1)

smodel = Splerge()
print("loading weights...")
smodel.load_state_dict(torch.load("model/full_train/model_192k.pth"))

model = Merge()

# print(model)
open("merge.json", "w").write(str(model))

for i, (image, target, img_path) in enumerate(train_loader):
    img_name = img_path[0].split("/")[-1][:-4]
    print("image:", img_name, image.shape)
    
    rpn, cpn = smodel(image)
    row_prob = rpn[2]
    col_prob = cpn[2]

    thresh = 0.7

    col_prob_img = utils.probs_to_image(col_prob, image.shape, axis=0)
    row_prob_img = utils.probs_to_image(row_prob, image.shape, axis=1)

    col_region = col_prob_img.detach().clone()
    col_region[col_region > thresh] = 1 
    col_region[col_region <= thresh] = 0
    col_region = (~col_region.bool()).float()

    row_region = row_prob_img.detach().clone()
    row_region[row_region > thresh] = 1
    row_region[row_region <= thresh] = 0
    row_region = (~row_region.bool()).float()    

    grid = utils.binary_grid_from_prob_images(row_prob_img, col_prob_img)
    # utils.tensor_to_numpy_image(col_region, write_path="merge_input/"+img_name+"_col.png")
    # utils.tensor_to_numpy_image(row_region, write_path="merge_input/"+img_name+"_row.png")
    utils.tensor_to_numpy_image(grid, write_path=img_name+"_grid.png")

    input_feature = torch.cat((image, 
                            row_prob_img, 
                            col_prob_img,
                            row_region, 
                            col_region, 
                            grid), 
                        1)
    # 1, 8, ?, ?
    print("Input feature:",input_feature.shape)
    # print(target[0].shape)
    outputs = model(input_feature)
    targets = None
    print(merge_loss(outputs, targets))
    # up_out, down_out, left_out, right_out = outputs
    # print(up_out[1])
    # print(down_out[1])
    # print(left_out[1])
    # print(right_out[1])

    if i == 0:
        break