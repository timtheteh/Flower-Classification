import plot_utils

"""
Batch size = 32
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
"""

d18 = plot_utils.read("../txt/unbuf_reg_resnet18.txt")
d50 = plot_utils.read("../txt/output_50.txt")
print("DONE")
d152 = plot_utils.read_time("../txt/output152.txt")

plot_utils.plot_resnets(d18, d50, d152, "loss")
plot_utils.plot_resnets(d18, d50, d152, "acc")


