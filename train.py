# Runs model training and validation, records results and saves checkpoints.
# ClearML import
from clearml import Task
from clearml import Logger

import torch
import torch.nn as nn
import torchvision.utils
import matplotlib.pyplot as plt
import copy
import os
from torcheval.metrics import MulticlassAccuracy

from leafapp.data import get_loaders
from leafapp.models.cnn import BasicCNN

# Initialise ClearML task
task = Task.init(
    project_name="Plant-Diseases_Image-Classification_CNN",
    task_name="Baseline CNN",
    output_uri=True
)

prefix = task.name+'_'+task.id+'/'
os.makedirs(prefix, exist_ok=True)

# Hyperparameters
config_params = {
    "learning_rate": 0.001, # optimiser's learning rate
    "batch_size": 32, # batch size for training
    "image_size": 128, # size of the input images
    "num_epochs": 4, # number of epochs to train for
}
task.connect(config_params)  # connects all hyperparameters to ClearML

# Device configuration
# Select GPU if available, else CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Image helper functions
# This takes an image tensor as an input and displays it using matplotlib.
def plot_plant_images(img, true_label=None, predicted_label=None):
    npimg = img.detach().cpu().permute(1, 2, 0).numpy()
    title = ""
    if true_label is not None:
        title = f"Label: {class_names[true_label]}"
    if predicted_label is not None:
        title += f". Predicted: {class_names[predicted_label]}"
    plt.title(title)
    plt.imshow(npimg)

# This takes a batch of image tensors and displays them in a grid using matplotlib.
def plot_plant_batch(img_batch, nrows=8):
    # create grid of images, with given number of rows
    img_grid = torchvision.utils.make_grid(img_batch, nrow=nrows)
    # the shape will be [3, 28*n_rows, 28*n_cols] where 3 stand for the RGB color channels  
    print(img_grid.shape) 
    # to make it work with plt.imshow, the channel dim must be the last
    img_grid = torch.permute(img_grid,[1,2,0])
    plt.title("Plant Images")
    plt.imshow(img_grid.numpy())  # convert to numpy array for matplotlib

# Load dataset
train_loader, val_loader, test_loader, class_names = get_loaders(
    image_size=config_params["image_size"],
    batch_size=config_params["batch_size"],
    seed=42,
    augment=True,
)

# Initialise the model and move it to the configured device
model = BasicCNN(
    num_classes=len(class_names),
    image_size=config_params["image_size"],
).to(device)

# Training and validation loop
def train_BasicCNN_model(model, loss_fn, train_loader, val_loader, device, config_params):
    n_epochs = config_params['num_epochs'] # fetch from experiment params
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config_params['learning_rate'])
    # accuracy metrics
    train_accuracy_fn = MulticlassAccuracy()
    val_accuracy_fn = MulticlassAccuracy()

    # keep track of best validation accuracy and best model
    best_accuracy = 0
    # make a deep copy of the model's weights because model.state_dict() contains references
    # (so they will change when the model is updated)
    best_model = copy.deepcopy(model)

    # Object used to log experimental values 
    logger = Logger.current_logger()

    # Training loop
    model.train()  # set model to training mode
    for epoch in range(n_epochs):
        model.train()  # set model to training mode
        for i, batch in enumerate(train_loader):
            inputs = batch["pixels"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            
            # Backward pass and step
            loss.backward()
            optimizer.step()

            # IMPORTANT: 
            # Now log training loss on this batch into ClearML by using the above created Logger object.
            # Call this series 'Training loss' 
            logger.report_scalar(
                title='Training loss', 
                series='loss',
                value=loss.item(),
                iteration=epoch * len(train_loader) + i
            )
            
            # use similar code to log training accuracy
            train_accuracy_fn.update(outputs, labels)
            logger.report_scalar(
                title='Training accuracy', 
                series='accuracy',
                value=train_accuracy_fn.compute().item(),
                iteration=epoch * len(train_loader) + i
            )
            train_accuracy_fn.reset()
            
        print(f'Epoch {epoch + 1} training complete')
        # save and upload model checkpoint to clearml server
        torch.save(model.state_dict(), prefix+f'basic-cnn-model-{epoch}.pt')

        # Validation phase
        model.eval()  # set model to evaluation mode
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                inputs = batch["pixels"].to(device)
                labels = batch["label"].to(device)

                outputs = model(inputs)
                val_accuracy_fn.update(outputs, labels)

            current_accuracy = val_accuracy_fn.compute()
            
            # log validation accuracy for this epoch
            logger.report_scalar(
                title='Validation accuracy', 
                series='accuracy',
                value=current_accuracy.item(),
                iteration=epoch * len(train_loader) + i
            )
            val_accuracy_fn.reset()
            # keep track of best validation accuracy and best model
            if current_accuracy>best_accuracy:
                best_accuracy=current_accuracy
                best_model = copy.deepcopy(model)
                print(f'Epoch {epoch + 1}, new best model found (validation accuracy: {best_accuracy})')
            print(f'Epoch {epoch + 1} validation complete')
    

    # also log the best accuracy (1 value only here, not a series)
    logger.report_single_value('best val accuracy', best_accuracy)

    return best_model

# Execute the training function
loss_fn = nn.CrossEntropyLoss()

best_model = train_BasicCNN_model(
    model, loss_fn, train_loader, val_loader, device, config_params)

# Once training ends, save the best model
# This will upload it into the clearml experiment UI
torch.save(best_model.state_dict(), prefix+'plant-disease-torch-best_model.pt')

# Logging debug samples to ClearML
model.eval()  # set model to evaluation mode
logger = Logger.current_logger()
for i in range(10):
    train_iter = iter(train_loader)
    batch = next(train_iter)

    # pick first entry of random batch
    input_img = batch["pixels"][0].to(device)
    true_label = batch["label"][0].to(device)

    with torch.no_grad():
        output = best_model(input_img.unsqueeze(0))  # add batch dimension
        predicted_label = torch.argmax(output)
        # in a grid of subplots of size 5 x 2, select the (i+1)-th subplot 
        plt.subplot(5,2,i+1)
        plot_plant_images(input_img, true_label.item(), predicted_label.item())
        
plt.tight_layout()

# If this wasn't logged automatically use (uncomment) the code below, which logs the matplotlib plot manually. 
# logger.report_matplotlib_figure(title="Debug Plant Disease sample", series='.', 
#                                 report_image=True, # if report_image=False it appears under the plots tab
#                                 figure=plt)