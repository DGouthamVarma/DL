import general_functions as gf 
from torchvision import models
import torch

def loadPretrainedModel():
	model = models.vgg16(pretrained = True)
	for param in model.parameters():
		param.requires_grad = False
	return model 

def infoFromUser(model):
	print("The features in the model are: ")
	for index, layer in enumerate(model.features):
		print("------------------------------------------------------------------")
		print("| " , index, " | ", layer)
	index = int(input("Choose the index of the layer for which you want to visualize the filters: "))
	# kernels = []
	# status = 1
	kernel = int(input("Enter the index of the kernel you want to visualize: "))
	# print("Enter the index of kernels you want to visualize. Type 0 to exit")
	# while status:
	# 	kernel_num = int(input())
	# 	if kernel_num==0:
	# 		break
	# 	else:
	# 		kernels.append(kernel_num)
	return index, kernel

def getKernelOutput(model, optimizer, input_image, epochs, index, kernel):
	for i in range(1, epochs+1):
		print("Epoch: ", i)
		optimizer.zero_grad()
		temp = input_image
		for idx, layer in enumerate(model.features):
			temp = layer(temp)
			if idx == index:
				break
		output = temp[0, kernel+1]
		loss = -torch.mean(output)
		print("Loss is ", loss)
		loss.backward()
		optimizer.step()
	return input_image

model = loadPretrainedModel()
index, kernel = infoFromUser(model)
print('You selected kernel ', kernel, 'in layer', model.features[index], 'to visualize')
input_image = gf.getRandomImage(150,180,(3,224,224))
preprocessed_image = gf.preprocessImage(input_image, requires_grad = True)
optimizer = torch.optim.Adam([preprocessed_image], lr=0.1, weight_decay=1e-6)
kernel_output = getKernelOutput(model, optimizer, preprocessed_image, 20, index, kernel)
reverse_processed_image = gf.getDeprocessedImage(kernel_output.detach())
gf.displayKernelOutput(reverse_processed_image)





