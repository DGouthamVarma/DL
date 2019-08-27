import torch
from torchvision import models
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

image = Image.open('multi.jpg')
image.show()

model = models.segmentation.deeplabv3_resnet101(pretrained = True)

model.eval()

transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(
		mean = [0.485, 0.456, 0.406],
		std = [0.229, 0.224, 0.225]
		)
	])

transformed_image = transform(image)

transformed_image = transformed_image.unsqueeze(0)

prediction = model(transformed_image)

semantic_masks = prediction['out']

output = torch.argmax(semantic_masks.squeeze(), dim = 0).detach().numpy()

label_colors = [(0, 0, 0),
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)]

r = np.zeros_like(output).astype(np.uint8)
g = np.zeros_like(output).astype(np.uint8)
b = np.zeros_like(output).astype(np.uint8)

for i in range(len(label_colors)):
	r[output == i] = label_colors[i][0]
	g[output == i] = label_colors[i][1]
	b[output == i] = label_colors[i][2]

semantic_segmented_image = np.stack([r,g,b], axis = 2)

plt.imshow(semantic_segmented_image)

plt.show()