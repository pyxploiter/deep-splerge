import os

import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

from transforms import get_transform
from dataloader import TableDataset

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
        #################### EXPERIMENT 2 ####################
        # x = x[0]
        #################### EXPERIMENT 2 ####################
        height, width = x.shape[-2:]
        x1 = F.relu(self.dil_conv2d_2(x))
        x2 = F.relu(self.dil_conv2d_3(x))
        x3 = F.relu(self.dil_conv2d_4(x))
        
        out_feature = torch.cat((x1, x2, x3), 1)

        if self.block_num < 4:
            out_feature = self.row_pool(out_feature)

        top_branch_x = self.row_conv_1x1_top(out_feature)
        top_branch_row_means = torch.mean(top_branch_x, dim=3)
        top_branch_proj_pools = top_branch_row_means.view(1,self.block_conv1x1_output, height,1).repeat(1,1,1,top_branch_x.shape[3])

        bottom_branch_x = self.row_conv_1x1_bottom(out_feature)
        bottom_branch_row_means = torch.mean(bottom_branch_x, dim=3)
        bottom_branch_proj_pools = bottom_branch_row_means.view(1,1, height,1).repeat(1,1,1,bottom_branch_x.shape[3])
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
        #################### EXPERIMENT 2 ####################
        # x = x[0]
        #################### EXPERIMENT 2 ####################
        height, width = x.shape[-2:]
        x1 = F.relu(self.dil_conv2d_2(x))
        x2 = F.relu(self.dil_conv2d_3(x))
        x3 = F.relu(self.dil_conv2d_4(x))
        
        out_feature = torch.cat((x1, x2, x3), 1)

        if self.block_num < 4:
            out_feature = self.col_pool(out_feature)

        top_branch_x = self.col_conv_1x1_top(out_feature)
        top_branch_row_means = torch.mean(top_branch_x, dim=2)
        top_branch_proj_pools = top_branch_row_means.view(1,self.block_conv1x1_output, 1, width).repeat(1,1,top_branch_x.shape[2],1)

        bottom_branch_x = self.col_conv_1x1_bottom(out_feature)
        bottom_branch_row_means = torch.mean(bottom_branch_x, dim=2)
        bottom_branch_proj_pools = bottom_branch_row_means.view(1,1,1,width).repeat(1,1,bottom_branch_x.shape[2],1)
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

        #################### EXPERIMENT 2 ####################
        # self.rpn_blocks = list()
        # self.rpn_blocks.append(RPN(block_num=1))
        # self.rpn_blocks.append(RPN(block_num=2))
        # self.rpn_blocks.append(RPN(block_num=3))
        # self.rpn_blocks.append(RPN(block_num=4))
        # self.rpn_blocks.append(RPN(block_num=5))
        # self.rpn_net = torch.nn.Sequential(*self.rpn_blocks)        

        # self.cpn_blocks = list()
        # self.cpn_blocks.append(CPN(block_num=1))
        # self.cpn_blocks.append(CPN(block_num=2))
        # self.cpn_blocks.append(CPN(block_num=3))
        # self.cpn_blocks.append(CPN(block_num=4))
        # self.cpn_blocks.append(CPN(block_num=5))
        # self.cpn_net = torch.nn.Sequential(*self.cpn_blocks)
        #################### EXPERIMENT 2 ####################
        
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

        #################### EXPERIMENT 2 ####################
        # rpn_x = self.sfcn(x)
        # cpn_x = rpn_x
        # cpn_x = self.sfcn(x)
        # rpn_x = self.rpn_net((rpn_x, None))
        # cpn_x = self.cpn_net((cpn_x, None))
        # print(rpn_x[0].shape, rpn_x[1].shape)
        #################### EXPERIMENT 2 ####################

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

def get_logits(sig_probs):
    """
    Arguments:
    ----------
    sig_probs: output sigmoid probs from model
    """

    pos = sig_probs.squeeze(dim=0).view(batch_size,sig_probs.shape[2],1)
    neg = torch.sub(1, sig_probs.squeeze(dim=0)).view(batch_size,sig_probs.shape[2],1)
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
    
    # print(r5)
    
    r3_logits = get_logits(r3)
    r4_logits = get_logits(r4)
    r5_logits = get_logits(r5)
    
    c3_logits = get_logits(c3)
    c4_logits = get_logits(c4)
    c5_logits = get_logits(c5)

    crit = torch.nn.BCELoss()
    rpn_targets = rpn_targets.float()
    cpn_targets = cpn_targets.float()
    # print(r3_logits.shape, rpn_targets.shape)
    rl3 = crit(r3, rpn_targets)
    rl4 = crit(r4, rpn_targets)
    rl5 = crit(r5, rpn_targets)

    cl3 = crit(c3, cpn_targets)
    cl4 = crit(c4, cpn_targets)
    cl5 = crit(c5, cpn_targets)

    # rl3 = cross_entropy_loss(r3_logits, rpn_targets)
    # rl4 = cross_entropy_loss(r4_logits, rpn_targets)
    # rl5 = cross_entropy_loss(r5_logits, rpn_targets)

    # cl3 = cross_entropy_loss(c3_logits, cpn_targets)
    # cl4 = cross_entropy_loss(c4_logits, cpn_targets)
    # cl5 = cross_entropy_loss(c5_logits, cpn_targets)

    rpn_loss = rl5 + (lambda4 * rl4) + (lambda3 * rl3)
    cpn_loss = cl5 + (lambda4 * cl4) + (lambda3 * cl3)
    
    # print("rpn_loss:", round(rpn_loss.item(),4), "cpn_loss:", round(cpn_loss.item(),4))

    loss = rpn_loss + cpn_loss

    # return rpn_loss
    return loss

def collate_fn(batch):
    return tuple(zip(*batch))

batch_size = 1
learning_rate = 0.0001

MODEL_STORE_PATH = 'model'

train_images_path = "data/images"
train_labels_path = "data/labels"

print("Loading dataset...")
dataset = TableDataset(os.getcwd(), train_images_path, train_labels_path, get_transform(train=True))

# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()

train_dataset = torch.utils.data.Subset(dataset, indices[:-20])
test_dataset = torch.utils.data.Subset(dataset, indices[-20:])

# define training and validation data loaders
train_loader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
   # collate_fn=collate_fn)

test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print("creating splerge model...")
model = Splerge().to(device)
# model = Splerge()
print(model)
# print(dir(model))
# print(model.cpn_block)
# for name, param in model.named_parameters():
#     print(name)
# exit(0)

criterion = splerge_loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
loss_list = []
acc_list = []

image, target = None, None

num_epochs = 300
train = True
evaluate = False

if train:
    model.train()
    print("starting training...")
    for epoch in range(num_epochs):
        if (epoch+1) % 100 == 0:
            torch.save(model.state_dict(), MODEL_STORE_PATH+'/model_ep{}.pth'.format(epoch+1))

        for i, (images, targets, img_path) in enumerate(train_loader):
            # print(img_path)
            images = images.to(device)
            
            targets[0] = targets[0].long().to(device)
            targets[1] = targets[1].long().to(device)
            
            image, target = images, targets

            # print("images:", images.shape)
            # print("targets", targets[0].shape)

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()        
            
            # Run the forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)

            loss_list.append(loss.item())

            loss.backward()
            optimizer.step()

            # for name, param in model.named_parameters():
            #     print(name, param.grad)

            # # Track the accuracy
            # total = labels.size(0)
            # _, predicted = torch.max(outputs.data, 1)
            # correct = (predicted == labels).sum().item()
            # acc_list.append(correct / total)

            if (i + 1) % 5 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))


############ EVALUATION ###############
if evaluate:
    # torch.save(model.state_dict(), MODEL_STORE_PATH+'/model_ep{}.pth'.format(epoch+1))
    model.load_state_dict(torch.load(MODEL_STORE_PATH+"/model_ep300.pth"))
    model.to(device)
    model.eval()

    with torch.no_grad():
        output = model(image)
        rpn_o, cpn_o = output

        loss = criterion(output, target)
        print(loss)

        r3, r4, r5 = rpn_o
        c3, c4, c5 = cpn_o

        r3, r4, r5 = r3[0][0], r4[0][0], r5[0][0] 
        c3, c4, c5 = c3[0][0], c4[0][0], c5[0][0]

        tr5 = target[0][0]
        tc5 = target[1][0]

        cout = c5.clone()
        cout[cout > 0.7] = 255
        cout[cout <= 0.7] = 0
        cout = cout.view(1, cout.shape[0]).repeat(cout.shape[0], 1)
        cout = cout.cpu().numpy()

        rout = r5.clone()
        rout[rout > 0.7] = 255
        rout[rout <= 0.7] = 0
        rout = rout.view(rout.shape[0], 1).repeat(1, rout.shape[0])
        rout = rout.cpu().numpy()
        
        tcout = tc5.view(1, tc5.shape[0]).repeat(tc5.shape[0], 1)
        tcout = tcout.cpu().numpy()
        tcout[tcout == 1] = 255

        trout = tr5.view(tr5.shape[0], 1).repeat(1, tr5.shape[0])
        trout = trout.cpu().numpy()
        trout[trout == 1] = 255

        import cv2
        cv2.imwrite("col_out.png", cout)
        cv2.imwrite("row_out.png", rout)
        cv2.imwrite("col.png", tcout)
        cv2.imwrite("row.png", trout)
        
        cv2.imshow("col_out", cout.astype("uint8"))
        cv2.imshow("col", tcout.astype("uint8"))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow("row_out", rout.astype("uint8"))
        cv2.imshow("row", trout.astype("uint8"))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # print(torch.abs(tr5 - r5))
        print(torch.sum(torch.abs(tr5 - r5)))
        print(torch.sum(torch.abs(tc5 - c5)))
