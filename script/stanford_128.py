import data_utils
import stanford_model_utils

train_loader, val_loader, test_loader = data_utils.load_data(128)


model = stanford_model_utils.MyModel()


name = "stanford_128"
stanford_model_utils.train_val(model, train_loader, val_loader, name, lr=0.01, gamma=0.9, num_epochs=40)
stanford_model_utils.test(model, test_loader, name)
