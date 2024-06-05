from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO

def load_image():
    getimage = int(input("Press 0 for local device image or 1 for online image: "))
    if getimage == 0:
        image_path = input("Enter path: ")
        return Image.open(image_path)
    else:
        url = input("Enter URL path in jpg format: ")
        return Image.open(requests.get(url, stream=True).raw)

def detect_objects(image):
    # Initialize the processor and model
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Convert outputs (bounding boxes and class logits) to COCO API format
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    # Collect detected objects with their labels and bounding boxes
    detected_objects = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if score > 0.9:
            box = [round(i, 2) for i in box.tolist()]
            label_name = model.config.id2label[label.item()]
            detected_objects.append((label_name, score.item(), box))
            print(f"Detected {label_name} with confidence {round(score.item(), 3)} at location {box}")
    return detected_objects

def generate_story(objects):
    if not objects:
        return "No significant objects were detected in the image."

    object_list = ', '.join([obj[0] for obj in objects])
    story_template = (
        f"In the image, I see {object_list}. "
        f"Each of these objects tells a part of the story. "
        f"As the sun rises, the {objects[0][0]} appears first, "
        f"followed by the {objects[1][0]} emerging from the shadows. "
        f"Together, they create a scene full of life and activity. "
        f"The {objects[2][0]} stands tall, watching over everything, "
        f"while the {objects[3][0]} moves gracefully across the frame. "
        f"Each element plays its part in this beautiful scene."
    )
    return story_template

# Main script execution
image = load_image()
detected_objects = detect_objects(image)
if detected_objects:
    story = generate_story(detected_objects)
    print("\nGenerated Story:")
    print(story)
else:
    print("No objects detected with sufficient confidence.")

1
