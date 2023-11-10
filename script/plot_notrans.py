".txt"
"unbuf_notrans_4_128.txt"

import plot_utils_deform

"""
Batch size = 32
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
"""

d3 = plot_utils_deform.read_val("../txt/unbuf_notrans_3_128.txt")
d4 = plot_utils_deform.read_val("../txt/unbuf_notrans_4_128.txt")

plot_utils_deform.plot_2(d3, d4, "loss")
plot_utils_deform.plot_2(d3, d4, "acc")


