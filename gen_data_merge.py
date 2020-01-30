from torch.utils.data import DataLoader
from transforms import get_transform
from dataloader import SplitTableDataset
import os, cv2
import utils
from split import SplitModel
import numpy as np
import pickle
import torch


print("Loading dataset...")
dataset = SplitTableDataset(os.getcwd(), "data/org_images", "data/labels", get_transform(train=True), False)
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
train_dataset = torch.utils.data.Subset(dataset, indices)
train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False, num_workers=1)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

smodel = SplitModel().to(device)
print("loading weights...")
smodel.load_state_dict(torch.load("model/model_625k.pth", map_location=device))

smodel.eval()

split_outs = {}

with torch.no_grad():
	for epoch in range(1):
	    for i, (image, target, img_path, W, H) in enumerate(train_loader):
	        img_name = img_path[0].split("/")[-1][:-4]
	        
	        # if not os.path.exists("data/split_labels/"+img_name+".pkl"):
	        print("["+str(i+1)+"/"+str(len(train_loader))+"]", img_name, "True")

	        rpn, cpn = smodel(image.to(device))

	        row_prob = rpn[2].cpu()
	        col_prob = cpn[2].cpu()

	        split_outs[img_name] = {"row_prob":row_prob, "col_prob":col_prob}
	        
with open("data/split_outs.pkl", "wb") as f:
    pickle.dump(split_outs, f)