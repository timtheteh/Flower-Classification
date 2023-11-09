import matplotlib.pyplot as plt

def read(file):
	with open(file, "r") as f:
		lines = f.readlines()
	
	epochs = []
	losses = []
	accuracies = []

	for i in range(0, len(lines)-2, 2):
		line = lines[i]
		parts = line.split(", ")
		epoch_part = parts[0]
		loss_part = parts[1]
		epoch_number = int(epoch_part.split(" ")[1])
		loss = float(loss_part.split(": ")[1].split("\n")[0])

		line = lines[i+1]
		accuracy = float(line.split(": ")[-1].strip("%\n"))

		epochs.append(epoch_number)
		losses.append(loss)
		accuracies.append(accuracy)
	
	test_acc = float(lines[-2].split(": ")[-1].strip("%\n"))
	test_loss = float(lines[-1].split(": ")[1].split("\n")[0])
	
	return {"epoch": epochs, "loss": losses, "acc": accuracies, "test": {"loss": test_loss, "acc": test_acc}}
	# return epochs, losses, accuracies


def read_time(file):
	with open(file, "r") as f:
		lines = f.readlines()
	
	epochs = []
	losses = []
	accuracies = []

	for i in range(0, len(lines)-3, 3):
		line = lines[i]
		epoch_number = int(line[6])
		parts = line.split(", ")
		epoch_part = parts[0]
		epoch_number = int(epoch_part.split(" ")[1])
		loss_part = parts[1]
		loss = float(loss_part.split(": ")[1])

		line = lines[i+1]
		accuracy = float(line.split(": ")[1].strip("%\n"))

		

		epochs.append(epoch_number)
		losses.append(loss)
		accuracies.append(accuracy)
	
	test_acc = float(lines[-2].split(": ")[-1].strip("%\n"))
	test_loss = float(lines[-1].split(": ")[1].split("\n")[0])
	
	return {"epoch": epochs, "loss": losses, "acc": accuracies, "test": {"loss": test_loss, "acc": test_acc}}
	# return epochs, losses, accuracies


def plot_resnets(d1, d2, d3, mode):
	if mode == "acc":
		y1 = d1["acc"]
		y2 = d2["acc"]
		y3 = d3["acc"]
	elif mode == "loss":
		y1 = d1["loss"]
		y2 = d2["loss"]
		y3 = d3["loss"]
	
	plt.plot(d1["epoch"], y1, label='Resnet-18', marker='.')
	plt.plot(d2["epoch"], y2, label='Resnet-50', marker='.')
	plt.plot(d3["epoch"], y3, label='Resnet-152', marker='.')

	plt.xlabel('Epoch')
	plt.ylabel(mode)
	plt.title(f'Training {mode} vs Epoch')
	plt.legend()

	plt.savefig('pretrained_resnet_'+mode+'_plot.png')
	plt.show()

	