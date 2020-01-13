import torch
import torch.nn.functional as F

class SFCN(torch.nn.Module):
    
    #Our batch shape for input x is (3,?,?)
    def __init__(self):
        super(SFCN, self).__init__()
        
        #Input channels = 3, output channels = 18
        self.conv1 = torch.nn.Conv2d(3, 18, kernel_size=7, stride=1, padding=3)

        self.conv2 = torch.nn.Conv2d(18, 18, kernel_size=7, stride=1, padding=3)

        self.dil_conv3 = torch.nn.Conv2d(18, 18, kernel_size=7, dilation=2, stride=1, padding=6)
        
    def forward(self, x):
        
        #Computes the activation of the first convolution
        #Size changes from (3,?,?) to (18,?,?)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.dil_conv3(x))

        return(x)

class RPN(torch.nn.Module):

    def __init__(self, block_num):
        super(RPN, self).__init__()

        self.block_num = block_num
        self.block_inputs = [18, 55, 55, 55, 55]
        self.block_conv1x1_output = 36
        self.block_output = 55

        self.dil_conv2d_2 = torch.nn.Conv2d(self.block_inputs[self.block_num-1], 6, kernel_size=5, dilation=2, stride=1, padding=4)
        self.dil_conv2d_3 = torch.nn.Conv2d(self.block_inputs[self.block_num-1], 6, kernel_size=5, dilation=3, stride=1, padding=6)
        self.dil_conv2d_4 = torch.nn.Conv2d(self.block_inputs[self.block_num-1], 6, kernel_size=5, dilation=4, stride=1, padding=8)

        # 1x2 max pooling for rows
        self.row_pool = torch.nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))

        # 1x1 convolution for top branch
        self.row_conv_1x1_top = torch.nn.Conv2d(18, self.block_conv1x1_output, kernel_size=1) 

        # 1x1 convolution for bottom branch
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
        # TODO: can be used unsqueeze(3) instead view
        top_branch_proj_pools = top_branch_row_means.view(batch_size,self.block_conv1x1_output, height,1).repeat(1,1,1,top_branch_x.shape[3])

        bottom_branch_x = self.row_conv_1x1_bottom(out_feature)
        bottom_branch_row_means = torch.mean(bottom_branch_x, dim=3)
        # TODO: can be used unsqueeze(3) instead view
        bottom_branch_proj_pools = bottom_branch_row_means.view(batch_size,1,height,1).repeat(1,1,1,bottom_branch_x.shape[3])
        bottom_branch_sig_probs = torch.sigmoid(bottom_branch_proj_pools)

        rpn_x = torch.cat((top_branch_proj_pools, out_feature, bottom_branch_sig_probs), 1)

        if self.block_num > 2:
            intermed_probs = bottom_branch_sig_probs[:, :, :, 0]
            return (rpn_x, intermed_probs)

        return (rpn_x, None)


class CPN(torch.nn.Module):

    def __init__(self, block_num):
        super(CPN, self).__init__()

        self.block_num = block_num
        self.block_inputs = [18, 55, 55, 55, 55]
        self.block_conv1x1_output = 36
        self.block_output = 55

        self.dil_conv2d_2 = torch.nn.Conv2d(self.block_inputs[self.block_num-1], 6, kernel_size=5, dilation=2, stride=1, padding=4)
        self.dil_conv2d_3 = torch.nn.Conv2d(self.block_inputs[self.block_num-1], 6, kernel_size=5, dilation=3, stride=1, padding=6)
        self.dil_conv2d_4 = torch.nn.Conv2d(self.block_inputs[self.block_num-1], 6, kernel_size=5, dilation=4, stride=1, padding=8)

        # 1x2 max pooling for rows
        self.col_pool = torch.nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))

        # 1x1 convolution for top branch
        self.col_conv_1x1_top = torch.nn.Conv2d(18, self.block_conv1x1_output, kernel_size=1) 

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
        # TODO: can be used unsqueeze(2) instead view
        top_branch_proj_pools = top_branch_col_means.view(batch_size,self.block_conv1x1_output, 1, width).repeat(1,1,top_branch_x.shape[2],1)

        bottom_branch_x = self.col_conv_1x1_bottom(out_feature)
        bottom_branch_col_means = torch.mean(bottom_branch_x, dim=2)
        # TODO: can be used unsqueeze(2) instead view
        bottom_branch_proj_pools = bottom_branch_col_means.view(batch_size,1,1,width).repeat(1,1,bottom_branch_x.shape[2],1)
        bottom_branch_sig_probs = torch.sigmoid(bottom_branch_proj_pools)

        cpn_x = torch.cat((top_branch_proj_pools, out_feature, bottom_branch_sig_probs), 1)
        
        if self.block_num > 2:
            intermed_probs = bottom_branch_sig_probs[:, :, 0, :]
            return (cpn_x, intermed_probs)

        return (cpn_x, None)

class Splerge(torch.nn.Module):
    
    #Our batch shape for input x is (3,?,?)
    def __init__(self):
        super(Splerge, self).__init__()
        self.blocks = 5
        self.sfcn = SFCN()
        
        self.rpn_block_1 = RPN(block_num=1)
        self.rpn_block_2 = RPN(block_num=2)
        self.rpn_block_3 = RPN(block_num=3)
        self.rpn_block_4 = RPN(block_num=4)
        self.rpn_block_5 = RPN(block_num=5)

        self.cpn_block_1 = CPN(block_num=1)
        self.cpn_block_2 = CPN(block_num=2)
        self.cpn_block_3 = CPN(block_num=3)
        self.cpn_block_4 = CPN(block_num=4)
        self.cpn_block_5 = CPN(block_num=5)

    def forward(self, x):
        # print("Input shape:", x.shape)

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

        return rpn_outputs, cpn_outputs