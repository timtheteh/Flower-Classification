import data_utils
import model_utils

train_loader, val_loader, test_loader = data_utils.load_data()


model = model_utils.MyModel(layer=2)


name = "trans_2"
model_utils.train_val(model, train_loader, val_loader, name, lr=0.01, gamma=0.9, num_epochs=40)
model_utils.test(model, test_loader, name)
