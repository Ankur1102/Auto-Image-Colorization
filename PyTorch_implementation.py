import torch.nn.functional as F
import torchvision.transforms as T
import torch.nn as nn
import torchvision.models as models
import torch
import numpy as np
from torchvision.datasets.folder import default_loader
from skimage.color import rgb2lab, rgb2gray, lab2rgb
from torchvision import datasets
import os
import matplotlib.pyplot as plt
import torchvision.transforms as T
import cv2
from torch.utils.tensorboard import SummaryWriter


# In this model, we tried to use a pretrained ResNet for feature extraction
# and then built a series of convolutional layers with batch normalization
# and upsampling to predict the AB channels. This way, we can leverage
# the power of transfer learning and improve our colorization results.

class ColorizationNet1(nn.Module):
    def __init__(self, input_channels=1):
        super(ColorizationNet1, self).__init__()

        # using a pretrained ResNet18 model for feature extraction
        # and modifying its first layer to accept grayscale images
        resnet = models.resnet18(num_classes=1000)
        resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1).unsqueeze(1))
        self.feature_extractor_resnet = nn.Sequential(*list(resnet.children())[:6])

        # main convolutional layers and batch normalization layers
        self.conv1 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 16, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(16)
        self.conv5 = nn.Conv2d(16, 2, 3, padding=1)

        # Defining the upsampling layers for increasing the spatial resolution
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.upsample3 = nn.Upsample(scale_factor=2)

    # Forward pass
    def forward(self, input):
        # First, we extract features using the modified ResNet
        features_resnet = self.feature_extractor_resnet(input)

        # Then, we pass the features through the main convolutional layers
        x = F.relu(self.bn1(self.conv1(features_resnet)))
        x = self.upsample1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.upsample2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.conv5(x)

        # Finally, we upsample the output to match the input size
        x = self.upsample3(x)

        return x



# here we're creating a custom dataset class that inherits from ImageFolder
class DatasetPrep(datasets.ImageFolder):

    # initializing the class, passing down any required arguments to the superclass
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader, is_train=True):
        super(DatasetPrep, self).__init__(root, transform=transform, target_transform=target_transform, loader=loader)
        self.is_train = is_train

    # Defining the __getitem__ method to fetch a specific item from the dataset
    def __getitem__(self, idx):
        img_path, _ = self.imgs[idx]
        img_data = self.loader(img_path)
        # Applying any specified transformations to the image
        img_data = self.transform(img_data)
        # converting the image to a NumPy array and rearranging the dimensions
        img_data = np.transpose(np.asarray(img_data), (1, 2, 0))
        # converting the RGB image to the LAB color space and normaklizing it
        lab_img = (rgb2lab(img_data) + 128) / 255
        # Extracting the AB channels and converting them to a PyTorch tensor
        ab_channels = torch.from_numpy(lab_img[:, :, 1:3].transpose((2, 0, 1))).float()
        grayscale_img = rgb2gray(img_data)
        grayscale_img = torch.from_numpy(grayscale_img).unsqueeze(0).float()
        # Returning the grayscale image and the AB channels as the output
        return grayscale_img, ab_channels

# a function to get data transforms for training or testing
def get_data_transforms(is_train=True):
    if is_train:
        return T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])
    else:
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
        ])
    

# main training
# we're defining a class for our image colorizer
class ImageColorizer:
    def __init__(self, writer):
        self.writer = writer
        self.train_loss_history = []
        self.val_loss_history = []

    # a helper method to convert LAB back to RGB
    @staticmethod
    def _convert_lab_to_rgb(l_channel, ab_channels):
        lab = torch.cat((l_channel, ab_channels), 0).numpy()
        lab = lab.transpose((1, 2, 0))
        lab[:, :, 0:1] *= 100
        lab[:, :, 1:3] = lab[:, :, 1:3] * 255 - 128
        return lab2rgb(lab.astype(np.float64))

    # This method will store the colorized images and their grayscale counterpart
    def _store_images(self, l_channel, ab_channels, path_dict, file_name):
        colored_image = self._convert_lab_to_rgb(l_channel, ab_channels)
        gray_image = l_channel.squeeze().numpy()
        plt.imsave(arr=gray_image, fname=f"{path_dict['grayscale']}{file_name}", cmap='gray')
        plt.imsave(arr=colored_image, fname=f"{path_dict['colorized']}{file_name}")

    # training loop for our colorizer
    def training(self, loader, epoch_idx, net, loss_function, optim):
        net.train()
        total_loss = 0
        for batch_idx, (l_input, ab_input) in enumerate(loader):
            l_input, ab_input = l_input.cuda(), ab_input.cuda()
            ab_output = net(l_input)
            loss = loss_function(ab_output, ab_input)
            total_loss += loss.item()
            optim.zero_grad()
            loss.backward()
            optim.step()
            self.writer.add_scalar("Training Loss", loss.item(), epoch_idx * len(loader) + batch_idx)

        average_loss = total_loss / len(loader)
        self.train_loss_history.append(average_loss)
        print(f'Epoch: [{epoch_idx + 1}] Average training Loss: {average_loss:.6f}')

    # The validation loop, it also saves some colorized images for visualization
    def validation(self, loader, epoch_idx, save_images_flag, net, loss_function):
        net.eval()
        total_loss = 0
        images_saved = False
        for batch_idx, (l_input, ab_input) in enumerate(loader):
            l_input, ab_input = l_input.cuda(), ab_input.cuda()
            ab_output = net(l_input)
            loss = loss_function(ab_output, ab_input)
            total_loss += loss.item()
            # We're checking if we should save images and if we haven't saved them yet
            if save_images_flag and not images_saved:
                # Setting the flag to True so that we only save one set of images
                images_saved = True
                # Defining the paths for saving grayscale, colorized, and ground truth images
                paths = {
                    'grayscale': '/content/drive/MyDrive/585_project_f04/outputs/gray/',
                    'colorized': '/content/drive/MyDrive/585_project_f04/outputs/color/',
                    'ground_truth': '/content/drive/MyDrive/585_project_f04/outputs/ground_truth/'
                }
                name = f'img-{batch_idx * loader.batch_size + 0}-epoch-{epoch_idx + 1}.jpg'
                # Storing the colorized and grayscale images using the helper method
                self._store_images(l_input[0].cpu(), ab_output[0].detach().cpu(), paths, name)
                gt_image = torch.cat((l_input[0].cpu(), ab_input[0].cpu()), 0).numpy()
                gt_image = gt_image.transpose((1, 2, 0))
                gt_image[:, :, 0:1] *= 100
                gt_image[:, :, 1:3] = gt_image[:, :, 1:3] * 255 - 128
                gt_image = lab2rgb(gt_image.astype(np.float64))
                # Saving the ground truth color image
                plt.imsave(arr=gt_image, fname=f"{paths['ground_truth']}{name}")
            self.writer.add_scalar("Validation Loss", loss.item(), epoch_idx * len(loader) + batch_idx)
        average_loss = total_loss / len(loader)
        self.val_loss_history.append(average_loss)
        print(f'Epoch: [{epoch_idx + 1}] Average validation Loss: {average_loss:.6f}')

    # it combines training and validation loops
    def train_and_validate(self, training_data_loader, validation_data_loader, epochs, net, loss_function, optim, save_images, T_max=None, eta_min=0):
        if T_max is None:
            T_max = epochs
        LRscheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=T_max, eta_min=eta_min, verbose=False)

        for epoch in range(epochs):
            self.training(training_data_loader, epoch, net, loss_function, optim)
            LRscheduler.step()
            with torch.no_grad():
                self.validation(validation_data_loader, epoch, save_images, net, loss_function)

# hyperparameters
epochs = 25
save_images = True
lr = 1e-3
weight_decay = 1e-4
save_model = True
loss = 'rmse'
batch_size = 32

# loading model
model = ColorizationNet1().cuda()

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, x, y):
        return torch.sqrt(nn.MSELoss()(x, y))
    
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

train_loss_history = []
val_loss_history = []

writer = SummaryWriter()
# checking which loss functions to use
if loss == 'mse':
    criterion = nn.MSELoss().cuda()
elif loss == 'rmse':
    criterion = RMSELoss().cuda()
elif loss == 'l1':
    criterion = nn.L1Loss().cuda()
else:
    print(f"Loss function: {loss} not defined")
    f= 1

# Training begins
if f ==1:
  print("please define the loss function or choose an existing loss function")
else:
  training_data_loader = torch.utils.data.DataLoader(DatasetPrep('/content/drive/MyDrive/585_project_f04/images_vs/train', get_data_transforms(is_train=True), is_train=True), batch_size= batch_size, shuffle=True)
  validation_data_loader = torch.utils.data.DataLoader(DatasetPrep('/content/drive/MyDrive/585_project_f04/images_vs/val', get_data_transforms(is_train=False), is_train=False), batch_size= batch_size, shuffle=False)
  colorizer = ImageColorizer(writer=writer)
  colorizer.train_and_validate(training_data_loader, validation_data_loader, epochs, model, criterion, optimizer, save_images)
# Saving model
if save_model:  
    torch.save(model, '/content/drive/MyDrive/585_project_f04/Models/saved_models/dn_32_31_rmse.pth')
