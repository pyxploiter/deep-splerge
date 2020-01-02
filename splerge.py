import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader


class SFCN(torch.nn.Module):
    
    #Our batch shape for input x is (3,256,256)
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

class Splerge(torch.nn.Module):
    
    #Our batch shape for input x is (3,256,256)
    def __init__(self):
        super(Splerge, self).__init__()
        self.blocks = 5
        self.block_inputs = [18, 55, 55, 55, 55]
        self.block_conv1x1_output = 36
        self.block_output = 55

        # get shared FCN
        self.sfcn = SFCN()

        #Input channels = 3, output channels = 18
        # self.dil_conv2 = torch.nn.Conv2d(18, 6, kernel_size=5, dilation=2, stride=1, padding=4)
        # self.dil_conv3 = torch.nn.Conv2d(18, 6, kernel_size=5, dilation=3, stride=1, padding=6)
        # self.dil_conv4 = torch.nn.Conv2d(18, 6, kernel_size=5, dilation=4, stride=1, padding=8)

        # 1x2 max pooling for rows
        self.row_pool = torch.nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
        # 2x1 max pooling for columns
        self.col_pool = torch.nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))

        # 1x1 convolution for top branch
        self.conv4_1x1_top = torch.nn.Conv2d(18, self.block_conv1x1_output, kernel_size=1) 

        # 1x1 convolution for bottom branch
        self.conv4_1x1_bottom = torch.nn.Conv2d(18, 1, kernel_size=1) 

    # dilated convolutions specifically for this RPN
    def dil_conv2d(self, input_feature, in_size, out_size=6, kernel_size=5, dilation=1, stride=1, padding=1):
        conv_layer = torch.nn.Conv2d(in_size, out_size, kernel_size=kernel_size, dilation=dilation, stride=stride, padding=padding)
        return conv_layer(input_feature)

    def rpn_block(self, input_feature, block_num):
        height, width = input_feature.shape[-2:]

        # dilated convolutions 2/3/4
        x1 = F.relu(self.dil_conv2d(input_feature, self.block_inputs[block_num-1], dilation=2, padding=4))
        x2 = F.relu(self.dil_conv2d(input_feature, self.block_inputs[block_num-1], dilation=3, padding=6))
        x3 = F.relu(self.dil_conv2d(input_feature, self.block_inputs[block_num-1], dilation=4, padding=8))
        # concatenating features
        out_feature = torch.cat((x1, x2, x3), 1)

        if block_num < 4:
            out_feature = self.row_pool(out_feature)

        # print("\nTop Branch:")
        top_branch_x = self.conv4_1x1_top(out_feature)
        # print("After 1x1 conv, shape:", top_branch_x.shape)
        top_branch_row_means = torch.mean(top_branch_x, dim=3)

        # print("Row means shape:", top_branch_row_means.shape)
        top_branch_proj_pools = top_branch_row_means.view(1,self.block_conv1x1_output, height,1).repeat(1,1,1,top_branch_x.shape[3])
        # print("After projection pooling:", top_branch_proj_pools.shape)

        # print("\nBottom Branch:")
        bottom_branch_x = self.conv4_1x1_bottom(out_feature)
        # print("After 1x1 conv, shape:", bottom_branch_x.shape)
        bottom_branch_row_means = torch.mean(bottom_branch_x, dim=3)
        # print("Row means shape:", bottom_branch_row_means.shape)
        bottom_branch_proj_pools = bottom_branch_row_means.view(1,1, height,1).repeat(1,1,1,top_branch_x.shape[3])
        # print("After projection pooling:", bottom_branch_proj_pools.shape)
        bottom_branch_sig_probs = torch.sigmoid(bottom_branch_proj_pools)
        # print("After sigmoid layer:", bottom_branch_sig_probs.shape)
        
        if block_num > 2:
            intermed_probs = bottom_branch_sig_probs[:,:,:,0]
            return (top_branch_proj_pools, 
                    bottom_branch_sig_probs, 
                    out_feature,
                    intermed_probs)

        return (top_branch_proj_pools, 
                bottom_branch_sig_probs,
                out_feature,
                None)

    def cpn_block(self, input_feature, block_num):
        height, width = input_feature.shape[-2:]

        # dilated convolutions 2/3/4
        x1 = F.relu(self.dil_conv2d(input_feature, self.block_inputs[block_num-1], dilation=2, padding=4))
        x2 = F.relu(self.dil_conv2d(input_feature, self.block_inputs[block_num-1], dilation=3, padding=6))
        x3 = F.relu(self.dil_conv2d(input_feature, self.block_inputs[block_num-1], dilation=4, padding=8))
        # concatenating features
        out_feature = torch.cat((x1, x2, x3), 1)

        if block_num < 4:
            out_feature = self.col_pool(out_feature)

        # print("\nTop Branch:")
        top_branch_x = self.conv4_1x1_top(out_feature)
        # print("After 1x1 conv, shape:", top_branch_x.shape)
        top_branch_row_means = torch.mean(top_branch_x, dim=2)

        # print("Row means shape:", top_branch_row_means.shape)
        top_branch_proj_pools = top_branch_row_means.view(1,self.block_conv1x1_output,1,width).repeat(1,1,top_branch_x.shape[2],1)
        # print("After projection pooling:", top_branch_proj_pools.shape)

        # print("\nBottom Branch:")
        bottom_branch_x = self.conv4_1x1_bottom(out_feature)
        # print("After 1x1 conv, shape:", bottom_branch_x.shape)
        bottom_branch_row_means = torch.mean(bottom_branch_x, dim=2)
        # print("Row means shape:", bottom_branch_row_means.shape)
        bottom_branch_proj_pools = bottom_branch_row_means.view(1,1,1,width).repeat(1,1,top_branch_x.shape[2],1)
        # print("After projection pooling:", bottom_branch_proj_pools.shape)
        bottom_branch_sig_probs = torch.sigmoid(bottom_branch_proj_pools)
        # print("After sigmoid layer:", bottom_branch_sig_probs.shape)
        
        if block_num > 2:
            intermed_probs = bottom_branch_sig_probs[:,:,1,:]
            return (top_branch_proj_pools, 
                    bottom_branch_sig_probs, 
                    out_feature,
                    intermed_probs)

        return (top_branch_proj_pools, 
                bottom_branch_sig_probs,
                out_feature,
                None)

    def forward(self, x):
        print("Input shape:", x.shape)
        
        rpn_x = self.sfcn(x)
        cpn_x = torch.tensor(rpn_x, requires_grad=True)

        rpn_outputs = []
        cpn_outputs = []
        for block_num in range(self.blocks):
            print("="*15,"BLOCK NUMBER:", block_num+1,"="*15)
            rpn_top, rpn_bottom, rpn_center, rpn_probs = self.rpn_block(input_feature=rpn_x, block_num=block_num+1)
            cpn_top, cpn_bottom, cpn_center, cpn_probs = self.cpn_block(input_feature=cpn_x, block_num=block_num+1)
            
            rpn_x = torch.cat((rpn_top, rpn_center, rpn_bottom), 1)
            cpn_x = torch.cat((cpn_top, cpn_center, cpn_bottom), 1)
            
            print("RPN output shape:", rpn_x.shape)
            print("CPN output shape:", cpn_x.shape)
            
            if rpn_probs is not None:
                rpn_outputs.append(rpn_probs)

            if cpn_probs is not None:
                cpn_outputs.append(cpn_probs)            

        return rpn_outputs, cpn_outputs

def get_logits(sig_probs):
    """
    Arguments:
    ----------
    sig_probs: output sigmoid probs from model
    """

    pos = sig_probs.squeeze(dim=0).view(input_dim,1)
    neg = torch.sub(1, sig_probs.squeeze(dim=0)).view(input_dim,1)
    logits = torch.cat((pos,neg),1)
    return logits

def cross_entropy_loss(logits, targets):
    """
    Arguments:
    ----------
    logits: (N, num_classes)
    targets: (N)
    """

    # print(torch.abs(x-y))
    log_prob = -1.0 * F.log_softmax(logits, 1)
    loss = log_prob.gather(1, targets.unsqueeze(1))
    loss = loss.mean()
    return loss

def splerge_loss(outputs, targets):
    """
    Arguments:
    ----------
    outputs: (r3, r4, r5)
    targets: (N)
    """
    
    lambda3 = 0.1
    lambda4 = 0.25

    r3, r4, r5 = outputs
    
    r3_logits = get_logits(r3)
    r4_logits = get_logits(r4)
    r5_logits = get_logits(r5)

    l3 = cross_entropy_loss(r3_logits, targets)
    l4 = cross_entropy_loss(r4_logits, targets)
    l5 = cross_entropy_loss(r5_logits, targets)

    loss = l5 + lambda4 * l4 + lambda3 * l3
    return loss

input_dim = 32
sample = torch.rand((1, 3, input_dim, input_dim))
sample_gt = torch.LongTensor(input_dim).random_(0,2)

model = Splerge()
rpn_outputs, cpn_outputs = model(sample)

criterion = splerge_loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

rpn_loss = criterion(rpn_outputs, sample_gt)
cpn_loss = criterion(cpn_outputs, sample_gt)

print(rpn_loss)
print(cpn_loss)

"""
num_epochs = 1
num_classes = 10
batch_size = 1
learning_rate = 0.001

DATA_PATH = 'data'
MODEL_STORE_PATH = 'model'

# transforms to apply to the data
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=True, transform=trans, download=True)
test_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=False, transform=trans)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model = RPN()
print(model)

# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
loss_list = []
acc_list = []

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Run the forward pass
        print(images.shape)
        outputs = model(images)
        print(outputs.shape)
        break
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))

"""