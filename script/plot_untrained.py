import plot_utils_deform

"""
Batch size = 32
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
"""

d50 = plot_utils_deform.read_val("../txt/unbuf_trans_3.txt")
d152 = plot_utils_deform.read_val("../txt/unbuf_trans_4_gpu.txt")
d18 = plot_utils_deform.read_val("../txt/trans_2.txt")

plot_utils_deform.plot_resnets(d18, d50, d152, "loss")
plot_utils_deform.plot_resnets(d18, d50, d152, "acc")


