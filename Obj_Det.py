from transformers import DetrImageProcessor, DetrForObjectDetection

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

from PIL import Image

# Load image
image = Image.open("R:\IS_Intern\sample images\1.webp")

# Preprocess image
inputs = processor(images=image, return_tensors="pt")

outputs = model(**inputs)

from transformers import DetrObjectDetectionPipeline

# Initialize object detection pipeline
detection_pipeline = DetrObjectDetectionPipeline(model=model, processor=processor)

# Run inference
results = detection_pipeline(image)

# Process results
boxes = results[0]["boxes"]
labels = results[0]["labels"]

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Plot image
plt.imshow(image)
ax = plt.gca()

# Draw bounding boxes
for box, label in zip(boxes, labels):
    xmin, ymin, xmax, ymax = box
    rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor="r", facecolor="none")
    ax.add_patch(rect)
    ax.text(xmin, ymin, processor.target_labels[label], fontsize=8, color="r")

plt.show()
