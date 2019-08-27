'''
	Some useful functions are defined using pytorch to easily import and use
	1. Preprocessing of an image for using transfer learning.
	2. Reverse processing the obtained output tensor
	3. Display multiple images as a grid
	4. Display an image
	5. Generating a randomly initialized image with in a range of integers
'''

#Importing the required libraries

import torch
from torchvision import transforms
import matplotlib.pyplot as plt 
import numpy as np
from torch.autograd import Variable

def normalizeImage(input_image):
	transform = transforms.Compose([
			transforms.Normalize(mean = [0.485, 0.456, 0.406],
				std = [0.229, 0.224, 0.225])
		])
	normalized_image = transform(input_image)
	return normalized_image

def getRandomImage(low, high, size):
	'''Returns a randomly generated image between low and high integers.
	All the arguments are required
	Arguments: 
	low - min integer value in the range of random integers
	high - max integer value in the range of random integers
	size - a tuple of dimensions required in order of (number_of_channels, height, width)
	'''
	generated_image = torch.randint(low, high, size, dtype=torch.float32)
	return generated_image


def preprocessImage(input_image, requires_grad):
	'''Returns the randomly initialized image after preprocessing for transfer learning with dimensions (1, channels, height, width)
	Arguments: 
	Randomly initialized image of dimensions (height, width, channels)
	A boolean value for requires_grad. True/False
	'''
	input_image = input_image/255
	normalized_image = normalizeImage(input_image)
	processed_image = normalized_image.unsqueeze(0)
	if requires_grad:
		processed_image = Variable(processed_image, requires_grad = True)
		return processed_image
	else:
		return processed_image


def getDeprocessedImage(kernel_output):
	mean = [-0.485, -0.456, -0.406]
	std = [1/0.229, 1/0.224, 1/0.225]
	kernel_output = kernel_output.squeeze(0)
	for i in range(3):
		kernel_output[i] /= std[i]
		kernel_output[i] -= mean[i]
	kernel_output[kernel_output>1] = 1
	kernel_output[kernel_output<0] = 0
	kernel_output = kernel_output * 255
	kernel_output = np.uint8(kernel_output)
	kernel_output = kernel_output.transpose(1,2,0)
	return kernel_output

def displayKernelOutput(input_image):
	plt.imshow(input_image)
	plt.show()

