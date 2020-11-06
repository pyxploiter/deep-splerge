import torch
import torch.nn.functional as F


class Split_SFCN(torch.nn.Module):
    def __init__(self):
        super(Split_SFCN, self).__init__()

        self.conv1 = torch.nn.Conv2d(3, 18, kernel_size=7, stride=1, padding=3)

        self.conv2 = torch.nn.Conv2d(18, 18, kernel_size=7, stride=1, padding=3)

        self.dil_conv3 = torch.nn.Conv2d(
            18, 18, kernel_size=7, dilation=2, stride=1, padding=6
        )

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.dil_conv3(x))

        return x


class Split_RPN(torch.nn.Module):
    def __init__(self, block_num):
        super(Split_RPN, self).__init__()

        self.block_num = block_num
        self.block_inputs = [18, 55, 55, 55, 55]
        self.block_conv1x1_output = 36
        self.block_output = 55

        self.dil_conv2d_2 = torch.nn.Conv2d(
            self.block_inputs[self.block_num - 1],
            6,
            kernel_size=5,
            dilation=2,
            stride=1,
            padding=4,
        )
        self.dil_conv2d_3 = torch.nn.Conv2d(
            self.block_inputs[self.block_num - 1],
            6,
            kernel_size=5,
            dilation=3,
            stride=1,
            padding=6,
        )
        self.dil_conv2d_4 = torch.nn.Conv2d(
            self.block_inputs[self.block_num - 1],
            6,
            kernel_size=5,
            dilation=4,
            stride=1,
            padding=8,
        )

        # 1x2 max pooling for rows
        self.row_pool = torch.nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        # 1x1 convolution for top branch
        self.row_conv_1x1_top = torch.nn.Conv2d(
            18, self.block_conv1x1_output, kernel_size=1
        )

        self.row_conv_1x1_bottom = torch.nn.Conv2d(18, 1, kernel_size=1)

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        x1 = F.relu(self.dil_conv2d_2(x))
        x2 = F.relu(self.dil_conv2d_3(x))
        x3 = F.relu(self.dil_conv2d_4(x))

        out_feature = torch.cat((x1, x2, x3), 1)

        if self.block_num < 4:
            out_feature = self.row_pool(out_feature)

        top_branch_x = self.row_conv_1x1_top(out_feature)
        top_branch_row_means = torch.mean(top_branch_x, dim=3)
        top_branch_proj_pools = top_branch_row_means.view(
            batch_size, self.block_conv1x1_output, height, 1
        ).repeat(1, 1, 1, top_branch_x.shape[3])

        bottom_branch_x = self.row_conv_1x1_bottom(out_feature)
        bottom_branch_row_means = torch.mean(bottom_branch_x, dim=3)
        bottom_branch_proj_pools = bottom_branch_row_means.view(
            batch_size, 1, height, 1
        ).repeat(1, 1, 1, bottom_branch_x.shape[3])
        bottom_branch_sig_probs = torch.sigmoid(bottom_branch_proj_pools)

        rpn_x = torch.cat(
            (top_branch_proj_pools, out_feature, bottom_branch_sig_probs), 1
        )

        if self.block_num > 2:
            intermed_probs = bottom_branch_sig_probs[:, :, :, 0]
            return (rpn_x, intermed_probs)

        return (rpn_x, None)


class Split_CPN(torch.nn.Module):
    def __init__(self, block_num):
        super(Split_CPN, self).__init__()

        self.block_num = block_num
        self.block_inputs = [18, 55, 55, 55, 55]
        self.block_conv1x1_output = 36
        self.block_output = 55

        self.dil_conv2d_2 = torch.nn.Conv2d(
            self.block_inputs[self.block_num - 1],
            6,
            kernel_size=5,
            dilation=2,
            stride=1,
            padding=4,
        )
        self.dil_conv2d_3 = torch.nn.Conv2d(
            self.block_inputs[self.block_num - 1],
            6,
            kernel_size=5,
            dilation=3,
            stride=1,
            padding=6,
        )
        self.dil_conv2d_4 = torch.nn.Conv2d(
            self.block_inputs[self.block_num - 1],
            6,
            kernel_size=5,
            dilation=4,
            stride=1,
            padding=8,
        )

        # 1x2 max pooling for rows
        self.col_pool = torch.nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        # 1x1 convolution for top branch
        self.col_conv_1x1_top = torch.nn.Conv2d(
            18, self.block_conv1x1_output, kernel_size=1
        )

        # 1x1 convolution for bottom branch
        self.col_conv_1x1_bottom = torch.nn.Conv2d(18, 1, kernel_size=1)

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        x1 = F.relu(self.dil_conv2d_2(x))
        x2 = F.relu(self.dil_conv2d_3(x))
        x3 = F.relu(self.dil_conv2d_4(x))

        out_feature = torch.cat((x1, x2, x3), 1)

        if self.block_num < 4:
            out_feature = self.col_pool(out_feature)

        top_branch_x = self.col_conv_1x1_top(out_feature)
        top_branch_col_means = torch.mean(top_branch_x, dim=2)
        top_branch_proj_pools = top_branch_col_means.view(
            batch_size, self.block_conv1x1_output, 1, width
        ).repeat(1, 1, top_branch_x.shape[2], 1)

        bottom_branch_x = self.col_conv_1x1_bottom(out_feature)
        bottom_branch_col_means = torch.mean(bottom_branch_x, dim=2)
        bottom_branch_proj_pools = bottom_branch_col_means.view(
            batch_size, 1, 1, width
        ).repeat(1, 1, bottom_branch_x.shape[2], 1)
        bottom_branch_sig_probs = torch.sigmoid(bottom_branch_proj_pools)

        cpn_x = torch.cat(
            (top_branch_proj_pools, out_feature, bottom_branch_sig_probs), 1
        )

        if self.block_num > 2:
            intermed_probs = bottom_branch_sig_probs[:, :, 0, :]
            return (cpn_x, intermed_probs)

        return (cpn_x, None)


class SplitModel(torch.nn.Module):
    def __init__(self, eval_mode=False):
        super(SplitModel, self).__init__()

        self.eval_mode = eval_mode
        self.blocks = 5
        self.sfcn = Split_SFCN()

        self.rpn_block_1 = Split_RPN(block_num=1)
        self.rpn_block_2 = Split_RPN(block_num=2)
        self.rpn_block_3 = Split_RPN(block_num=3)
        self.rpn_block_4 = Split_RPN(block_num=4)
        self.rpn_block_5 = Split_RPN(block_num=5)

        self.cpn_block_1 = Split_CPN(block_num=1)
        self.cpn_block_2 = Split_CPN(block_num=2)
        self.cpn_block_3 = Split_CPN(block_num=3)
        self.cpn_block_4 = Split_CPN(block_num=4)
        self.cpn_block_5 = Split_CPN(block_num=5)

    def forward(self, x):

        x = self.sfcn(x)

        rpn_x, _ = self.rpn_block_1(x)
        rpn_x, _ = self.rpn_block_2(rpn_x)
        rpn_x, rpn_probs_1 = self.rpn_block_3(rpn_x)
        rpn_x, rpn_probs_2 = self.rpn_block_4(rpn_x)
        rpn_x, rpn_probs_3 = self.rpn_block_5(rpn_x)

        rpn_outputs = [rpn_probs_1, rpn_probs_2, rpn_probs_3]

        cpn_x, _ = self.cpn_block_1(x)
        cpn_x, _ = self.cpn_block_2(cpn_x)
        cpn_x, cpn_probs_1 = self.cpn_block_3(cpn_x)
        cpn_x, cpn_probs_2 = self.cpn_block_4(cpn_x)
        cpn_x, cpn_probs_3 = self.cpn_block_5(cpn_x)

        cpn_outputs = [cpn_probs_1, cpn_probs_2, cpn_probs_3]

        if self.eval_mode:
            return rpn_outputs[2], cpn_outputs[2]

        return rpn_outputs, cpn_outputs


class Merge_SFCN(torch.nn.Module):
    def __init__(self):
        super(Merge_SFCN, self).__init__()

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

class Merge_Block(torch.nn.Module):

    def __init__(self, block_num):
        super(Merge_Block, self).__init__()

        self.block_num = block_num
        self.block_inputs = [32, 55, 55]
        self.block_conv1x1_output = [36,36,64]
        self.block_output = 55

        self.dil_conv2d_1 = torch.nn.Conv2d(self.block_inputs[self.block_num-1], 6, kernel_size=5, dilation=1, stride=1, padding=2)
        self.dil_conv2d_2 = torch.nn.Conv2d(self.block_inputs[self.block_num-1], 6, kernel_size=5, dilation=2, stride=1, padding=4)
        self.dil_conv2d_3 = torch.nn.Conv2d(self.block_inputs[self.block_num-1], 6, kernel_size=5, dilation=3, stride=1, padding=6)

        # 1x1 convolution for top branch
        self.conv_1x1_top = torch.nn.Conv2d(18, self.block_conv1x1_output[self.block_num-1], kernel_size=1) 

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
        
        np_grid = grid.cpu().numpy()
        r_mids, c_mids = utils.get_midpoints_from_grid(np_grid)
        row_mids.extend(r_mids)
        row_mids.append(grid.shape[0])
        col_mids.extend(c_mids)
        col_mids.append(grid.shape[1])

        for row_id in range(1, len(row_mids)):
            for col_id in range(1, len(col_mids)):              
                grid_mean = torch.mean(top_branch_x[:, :, row_mids[row_id-1]:row_mids[row_id], col_mids[col_id-1]: col_mids[col_id]]
                                        .reshape(batch_size,self.block_conv1x1_output[self.block_num-1],-1), dim=2)

                top_branch_x[:, :, row_mids[row_id-1]:row_mids[row_id], col_mids[col_id-1]: col_mids[col_id]] = grid_mean.unsqueeze(2).unsqueeze(2)
                
        ################# BOTTOM BRANCH #######################
        bottom_branch_x = self.conv_1x1_bottom(out_feature)

        temp = torch.tensor((), dtype=torch.float32)
        bottom_branch_grid_pools = temp.new_zeros((batch_size, 1, len(row_mids)-1, len(col_mids)-1))

        # for row_id in range(1, len(row_mids)):
        #     for col_id in range(1, len(col_mids)):
        #         grid_mean = torch.mean(bottom_branch_x[:, :, row_mids[row_id-1]:row_mids[row_id], col_mids[col_id-1]: col_mids[col_id]])
        #         bottom_branch_x[:, :, row_mids[row_id-1]:row_mids[row_id], col_mids[col_id-1]: col_mids[col_id]] = grid_mean
        #         bottom_branch_grid_pools[:, 0, row_id-1, col_id-1] = torch.sigmoid(grid_mean)
        # bottom_branch_sig_probs = torch.sigmoid(bottom_branch_x)
        # out_x = torch.cat((top_branch_x, out_feature, bottom_branch_sig_probs), 1)

        # bottom_branch_x_sig = torch.sigmoid(bottom_branch_x)
        for row_id in range(1, len(row_mids)):
            for col_id in range(1, len(col_mids)):
                grid_mean = torch.mean(torch.sigmoid(bottom_branch_x[:, :, row_mids[row_id-1]:row_mids[row_id], col_mids[col_id-1]: col_mids[col_id]]))
                bottom_branch_x[:, :, row_mids[row_id-1]:row_mids[row_id], col_mids[col_id-1]: col_mids[col_id]] = grid_mean
                bottom_branch_grid_pools[:, 0, row_id-1, col_id-1] = grid_mean
                
        out_x = torch.cat((top_branch_x, out_feature, bottom_branch_x), 1)

        if self.block_num > 1:
            return (out_x, bottom_branch_grid_pools)

        return (out_x, None)


class MergeModel(torch.nn.Module):

    def __init__(self):
        super(MergeModel, self).__init__()
        self.blocks = 3
        self.sfcn = Merge_SFCN()

        self.grid = None
        
        self.up_block_1 = Merge_Block(block_num=1)
        self.up_block_2 = Merge_Block(block_num=2)
        self.up_block_3 = Merge_Block(block_num=3)

        self.down_block_1 = Merge_Block(block_num=1)
        self.down_block_2 = Merge_Block(block_num=2)
        self.down_block_3 = Merge_Block(block_num=3)

        self.left_block_1 = Merge_Block(block_num=1)
        self.left_block_2 = Merge_Block(block_num=2)
        self.left_block_3 = Merge_Block(block_num=3)

        self.right_block_1 = Merge_Block(block_num=1)
        self.right_block_2 = Merge_Block(block_num=2)
        self.right_block_3 = Merge_Block(block_num=3)

    def forward(self, x):
        self.grid = x[:,7].clone().detach()
        self.grid = self.grid.squeeze(0).squeeze(0)

        features = self.sfcn(x)

        up, _ = self.up_block_1(features, self.grid)
        up, up_probs_1 = self.up_block_2(up, self.grid)
        up, up_probs_2 = self.up_block_3(up, self.grid)

        down, _ = self.down_block_1(features, self.grid)
        down, down_probs_1 = self.down_block_2(down, self.grid)
        down, down_probs_2 = self.down_block_3(down, self.grid)

        D1_probs = (1/2)*(up_probs_1[:, :, 1:, :] * down_probs_1[:, :, :-1, :]) + (1/4)*(up_probs_1[:,:,1:,:]+down_probs_1[:,:,:-1,:])
        D2_probs = (1/2)*(up_probs_2[:, :, 1:, :] * down_probs_2[:, :, :-1, :]) + (1/4)*(up_probs_2[:,:,1:,:]+down_probs_2[:,:,:-1,:])

        left, _ = self.left_block_1(features, self.grid)
        left, left_probs_1 = self.left_block_2(left, self.grid)
        left, left_probs_2 = self.left_block_3(left, self.grid)

        right, _ = self.right_block_1(features, self.grid)
        right, right_probs_1 = self.right_block_2(right, self.grid)
        right, right_probs_2 = self.right_block_3(right, self.grid)

        R1_probs = (1/2)*(left_probs_1[:, :, :, 1:] * right_probs_1[:, :, :, :-1]) + (1/4)*(left_probs_1[:,:,:,1:]+right_probs_1[:,:,:,:-1])
        R2_probs = (1/2)*(left_probs_2[:, :, :, 1:] * right_probs_2[:, :, :, :-1]) + (1/4)*(left_probs_2[:,:,:,1:]+right_probs_2[:,:,:,:-1])
                        
        return D1_probs, D2_probs, R1_probs, R2_probs
