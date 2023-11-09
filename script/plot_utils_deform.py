import matplotlib.pyplot as plt



def read_val(file):
	with open(file, "r") as f:
		lines = f.readlines()
	
	epochs = []
	losses = []
	accuracies = []

	for i in range(0, len(lines), 3):
		line = lines[i]
		epoch_number = int(line[6])
		parts = line.split(", ")
		epoch_part = parts[0]
		epoch_number = int(epoch_part.split(" ")[1])

		line = lines[i+2]
		loss = float(line.split(": ")[1].strip("%\n"))

		line = lines[i+1]
		accuracy = float(line.split(": ")[1].strip("%\n"))

		epochs.append(epoch_number)
		losses.append(loss)
		accuracies.append(accuracy)
	
	
	return {"epoch": epochs, "loss": losses, "acc": accuracies}
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
	
	plt.plot(d1["epoch"], y1, label='2 groups', marker='.')
	plt.plot(d2["epoch"], y2, label='3 groups', marker='.')
	plt.plot(d3["epoch"], y3, label='4 groups', marker='.')

	plt.xlabel('Epoch')
	if mode == "acc":
		plt.ylabel('Accuracy (%)')
	elif mode == "loss":
		plt.ylabel('Loss')
	plt.title(f'Validation {mode} vs Epoch')
	plt.legend()

	plt.savefig('untrained_trans_deform_'+mode+'_plot.png')
	plt.show()

	