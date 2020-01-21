import os
import argparse

import cv2
import numpy as np
import torch

import utils
from split import SplitModel

parser = argparse.ArgumentParser()

parser.add_argument("-p", "--test_images_dir", dest="test_images_dir", help="Path to training data images.", default="test_images")
parser.add_argument("-m","--model_weights", dest="model_weights", help="path to model weights.", default="model/model.pth")
parser.add_argument("-s","--output_path", dest="output_path", help="path to the output directory", default="outputs")

configs = parser.parse_args()

if not os.path.exists(configs.output_path):
	os.mkdir(configs.output_path)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print("creating split model...")
model = SplitModel(eval_mode=True).to(device)

print("loading weights...")
model.load_state_dict(torch.load(configs.model_weights, map_location=device))

model.eval()

images = os.listdir(configs.test_images_dir)

print("Predicting table rows and columns:")
print(40*"-")
for i, image_name in enumerate(images):
	print("["+str(i+1)+"/"+str(len(images))+"]", image_name)
	image_path = os.path.join(configs.test_images_dir, image_name)
	image = cv2.imread(image_path)
	H, W, C = image.shape
	image_trans = image.transpose((2,0,1)).astype('float32')
	resized_image = utils.resize_image(image_trans)
	input_image = utils.normalize_numpy_image(resized_image).unsqueeze(0)
	
	rpn_out, cpn_out = model(input_image.to(device))

	rpn_image = utils.probs_to_image(rpn_out.detach().clone(), input_image.shape, 1)
	cpn_image = utils.probs_to_image(cpn_out.detach().clone(), input_image.shape, 0)

	grid_img, row_image, col_image = utils.binary_grid_from_prob_images(rpn_image, cpn_image)

	grid_np_img = utils.tensor_to_numpy_image(grid_img)

	grid_np_img = cv2.resize(grid_np_img, (W,H))
	grid_np_img = cv2.cvtColor(grid_np_img, cv2.COLOR_GRAY2BGR)
	test_image = image.copy()
	test_image[np.where((grid_np_img == [255, 255, 255]).all(axis = 2))] = [0, 255, 0]
	cv2.imwrite(os.path.join(configs.output_path, image_name[:-4]+".png"), test_image)

	row_img = image.copy()
	rpn_image[rpn_image > 0.7] = 255
	rpn_image[rpn_image <= 0.7] = 0
	rpn_image = rpn_image.squeeze(0).squeeze(0).detach().numpy()
	rpn_image = cv2.resize(rpn_image, (W,H), interpolation=cv2.INTER_NEAREST)
	rpn_image = cv2.cvtColor(rpn_image, cv2.COLOR_GRAY2BGR)
	row_img[np.where((rpn_image == [255, 255, 255]).all(axis = 2))] = [255, 0, 255]
	cv2.imwrite(os.path.join(configs.output_path, image_name[:-4]+"_row.png"), row_img)

	col_img = image.copy()
	cpn_image[cpn_image > 0.7] = 255
	cpn_image[cpn_image <= 0.7] = 0
	cpn_image = cpn_image.squeeze(0).squeeze(0).detach().numpy()
	cpn_image = cv2.resize(cpn_image, (W,H), interpolation=cv2.INTER_NEAREST)
	cpn_image = cv2.cvtColor(cpn_image, cv2.COLOR_GRAY2BGR)
	col_img[np.where((cpn_image == [255, 255, 255]).all(axis = 2))] = [255, 0, 255]
	cv2.imwrite(os.path.join(configs.output_path, image_name[:-4]+"_col.png"), col_img)
