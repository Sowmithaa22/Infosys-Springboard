import urllib.request 
from transformers import pipeline
from PIL import Image, ImageDraw, ImageFont


font = ImageFont.truetype("arial.ttf", 25)

object_detector = pipeline("object-detection")

def draw_box(im, label, xmin, ymin, xmax, ymax, index, num_boxes):
	print(f"Drawing bounding box {index} of {num_boxes}...")

	im_with_rectangle = ImageDraw.Draw(im)  
	im_with_rectangle.rounded_rectangle((xmin, ymin, xmax, ymax), outline = "purple", width = 3, radius = 8)
	im_with_rectangle.text((xmin+35, ymin-25), label, fill="white", stroke_fill = "red", font = font)

	return im

def open_image(image):
	bounding_boxes = object_detector(image)
	num_boxes = len(bounding_boxes)
	index = 0

	for bounding_box in bounding_boxes:
		box = bounding_box["box"]
		image = draw_box(image, bounding_box["label"], box["xmin"], box["ymin"], box["xmax"], box["ymax"], index, num_boxes)
		index += 1
	return image.show("image.png")
		
	

user_input = int(input("\n\n\n Image Source:, \n 1) internal image from folder\n 2) Image URL from web\n Enter 1 or 2 : "))
if user_input == 1:
	internal_image = input('Internal image name or path: ')
	with Image.open(internal_image) as image:
		open_image(image)
		print("Open generated image with bounding boxes")
		
else:
	internal_image = input('Enter image url here : ')
	urllib.request.urlretrieve(internal_image, "image.png")
	with Image.open("image.png") as image:
		open_image(image)
		print("Image with bounding boxes is generated.")
