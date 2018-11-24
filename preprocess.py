#!/usr/bin/env python

from PIL import Image

import numpy as np
import os
import sys
import json

raw_data_path = 'data/raw/'
gt_data_path = 'data/gt/'
raw_output_path = 'npy/raw/'
gt_output_path = 'npy/gt/'
img_info = []
gt_info = []

def read_image():
	max_height = max_width = max_size = 0
	min_height = min_width = min_size = 2 ** 32 - 1	

	try:
		os.mkdir(raw_output_path)
		os.mkdir(gt_output_path)
	except:
		pass

	images = os.listdir(raw_data_path)
	for filename in images:
		if not (filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png')): continue
		info = {}
		info["name"] = filename
		tokens = filename.split(".")
		img = Image.open(raw_data_path + filename)
		img_np = np.asarray(img)
		height, width, _ = img_np.shape
		info["size"] = {
			"height": height,
			"width": width
		}			
		np.save(os.path.join(raw_output_path, tokens[0] + '.npy'), img_np)
		img_info.append(info)
		max_height = max(max_height, height)
		min_height = min(min_height, height)
		max_width = max(max_width, width)
		min_width = min(min_width, width)
		max_size = max(max_size, height * width)
		min_size = min(min_size, height * width)

	print("Height max/min:", max_height, min_height)
	print("Width max/min:", max_width, min_width)
	print("Size max/min:", max_size, min_size)

	with open('general', 'w') as f:
		f.write("Height max/min: {0} {1}\n".format(max_height, min_height))
		f.write("Width max/min: {0} {1}\n".format(max_width, min_width))
		f.write("Size max/min: {0} {1}\n".format(max_size, min_size))

	with open('image_info.json', 'w') as f:
		f.write(json.dumps(img_info))

	gts = os.listdir(gt_data_path)
	for filename in gts:
		info = {}
		info["name"] = filename
		if not (filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png')): continue
		tokens = filename.split(".")
		img = Image.open(gt_data_path + filename)
		img_np = np.asarray(img)
		# png binary image only 0 or 1
		height, width = img_np.shape
		np.save(os.path.join(gt_output_path, tokens[0] + '.npy'), img_np)
		info["size"] = {
			"height": height,
			"width": width
		}			
		gt_info.append(info)

	with open('gt_info.json', 'w') as f:
		f.write(json.dumps(gt_info))

if __name__ == '__main__':
	read_image()