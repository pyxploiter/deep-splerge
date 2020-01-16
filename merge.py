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
		
		np_grid = grid.cpu().numpy()
		r_mids, c_mids = utils.get_midpoints_from_grid(np_grid)
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
		bottom_branch_grid_pools = temp.new_zeros((batch_size, 1, len(row_mids)-1, len(col_mids)-1))

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

		self.grid = None
		
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

def merge_loss(outputs, targets):
	D1_probs, D2_probs, R1_probs, R2_probs = outputs

	td = torch.FloatTensor([[0,0,0,0],[1,1,1,1]])
	tr = torch.FloatTensor([[0,0,1],[0,0,1],[0,0,1]])

	targets = [td.unsqueeze(0), tr.unsqueeze(0)]
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
import torchviz


print("Loading dataset...")
dataset = TableDataset(os.getcwd(), "data/images", "data/labels", get_transform(train=True), False)
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
train_dataset = torch.utils.data.Subset(dataset, indices)
train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False, num_workers=1)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
smodel = Splerge().to(device)
print("loading weights...")
smodel.load_state_dict(torch.load("model/full_train/model_192k.pth"))

model = Merge().to(device)

open("split.json", "w").write(str(smodel))
open("merge.json", "w").write(str(model))

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=options.decay_rate)


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

model.train()
for epoch in range(500):
	for i, (image, target, img_path) in enumerate(train_loader):
		if i < 15: continue

		if i > 15: break

		img_name = img_path[0].split("/")[-1][:-4]
		
		rpn, cpn = smodel(image.to(device))
		row_prob = rpn[2].cpu()
		col_prob = cpn[2].cpu()

		thresh = 0.7

		col_prob_img = utils.probs_to_image(col_prob.detach().clone(), image.shape, axis=0)
		row_prob_img = utils.probs_to_image(row_prob.detach().clone(), image.shape, axis=1)

		col_region = col_prob_img.detach().clone()
		col_region[col_region > thresh] = 1 
		col_region[col_region <= thresh] = 0
		col_region = (~col_region.bool()).float()

		row_region = row_prob_img.detach().clone()
		row_region[row_region > thresh] = 1
		row_region[row_region <= thresh] = 0
		row_region = (~row_region.bool()).float()    

		grid_img = utils.binary_grid_from_prob_images(row_prob_img, col_prob_img)

		input_feature = torch.cat((image, 
								row_prob_img, 
								col_prob_img,
								row_region, 
								col_region, 
								grid_img), 
							1)

		# writer.add_graph(model, input_feature.to(device))

		optimizer.zero_grad()
		outputs = model(input_feature.to(device))

		targets = None
		loss, d_loss, r_loss = merge_loss(outputs, targets)
		print(loss.item(), d_loss.item(), r_loss.item())
		loss.backward()
		optimizer.step()

		torch.cuda.empty_cache()

		