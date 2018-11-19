#!/usr/bin/env python

from PIL import Image

import numpy as np
import os
import sys

data_path = 'data/'
output_path = 'npy/'
img_info = []

def read_image():
	max_height = max_width = max_size = 0
	min_height = min_width = min_size = 2 ** 32 - 1
	images = os.listdir(data_path)
	try:
		os.mkdir(output_path)
	except:
		pass

	for filename in images:
		if not (filename.endswith('.jpg') or filename.endswith('.jpeg')): continue
		info = {}
		info["name"] = filename
		tokens = filename.split(".")
		img = Image.open(data_path + filename)
		img_np = np.asarray(img)
		height, width, _ = img_np.shape
		info["size"] = {
			"height": height,
			"width": width
		}			
		np.save(os.path.join(output_path, tokens[0] + '.npy'), img_np)
		img_info.append(info)
		max_height = max(max_height, height)
		min_height = min(min_height, height)
		max_width = max(max_width, width)
		min_width = min(min_width, width)
		max_size = max(max_size, height * width)
		min_size = min(min_size, height * width)

	print(max_height, min_height)
	print(max_width, min_width)
	print(max_size, min_size)

if __name__ == '__main__':
	read_image()