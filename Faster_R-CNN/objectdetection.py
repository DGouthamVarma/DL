import torch
from torchvision import transforms as T
from torchvision import models
from PIL import Image
import cv2
import matplotlib.pyplot as plt


category_names = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

model = models.detection.fasterrcnn_resnet50_fpn(pretrained = True)

model.eval()

image = Image.open('image.jpg')
image.show()
transform = T.Compose([T.ToTensor()])
transformed_image = transform(image)

prediction = model([transformed_image])

predicted_scores = [i for i in prediction[0]['scores'].detach().numpy()]
bounding_boxes = [[(i[0], i[1]),(i[2], i[3])] for i in list(prediction[0]['boxes'].detach().numpy())]
predicted_classes = [category_names[i] for i in prediction[0]['labels']]

prediction_threshold = [predicted_scores.index(x) for x in predicted_scores if x > 0.5]
predicted_classes = predicted_classes[:len(prediction_threshold)]
bounding_boxes = bounding_boxes[:len(prediction_threshold)]

img = cv2.imread('image.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

for i in range(len(bounding_boxes)):
    cv2.rectangle(img, bounding_boxes[i][0], bounding_boxes[i][1], color = (0,255,0), thickness = 1)
    cv2.putText(img, predicted_classes[i], bounding_boxes[i][0], fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 0.5, color = (255, 0,0), thickness = 1)
    
plt.figure(figsize=(50,50))
plt.imshow(img)
plt.show()








