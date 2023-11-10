from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def image_trans(input_str):
	input_path = "../sample_images/" + input_str
	input_image = Image.open(input_path)
	# Define the transformations
	transform = transforms.Compose([
			transforms.RandomResizedCrop(224),
			transforms.RandomHorizontalFlip(),
			transforms.RandomRotation(30),
			transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
			transforms.RandomGrayscale(p=0.2),
			transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
			transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
			transforms.RandomVerticalFlip(),
	])
	# Apply the transformations to the input image
	output_image = transform(input_image)
	output_path = "../sample_images/trans_" + input_str
	output_image.save(output_path)
	return output_path


def grid_4(image_paths):
	# Create a figure with a 2x2 grid of subplots
	fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
	plt.subplots_adjust(wspace=0.06, hspace=0.06)
	# Loop through images and display them in the subplots
	for i, ax in enumerate(axes.flat):
		if image_paths[i][0] != ".":
			image_paths[i] = "../sample_images/" + image_paths[i]
		img = mpimg.imread(image_paths[i])
		ax.imshow(img)
		ax.axis("off")
	# Show the grid of images
	plt.show()

input_lst = ["image_00003.jpg", "image_00400.jpg", "image_01927.jpg", "image_04545.jpg"]
output_lst = []
for ele in input_lst:
	out = image_trans(ele)
	output_lst.append(out)

grid_4(input_lst)
grid_4(output_lst)

