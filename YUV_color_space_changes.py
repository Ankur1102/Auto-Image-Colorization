import numpy as np
from skimage.color import rgb2yuv, yuv2rgb
from torchvision import datasets
import torch
from torchvision.datasets.folder import default_loader
import matplotlib.pyplot as plt

# using YUV color space, separating Y channel and UV channels
class DatasetPrepYUV(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader, is_train=True):
        super(DatasetPrepYUV, self).__init__(root, transform=transform, target_transform=target_transform, loader=loader)
        self.is_train = is_train

    def __getitem__(self, idx):
        img_path, _ = self.imgs[idx]
        img_data = self.loader(img_path)
        img_data = self.transform(img_data)
        img_data = np.transpose(np.asarray(img_data), (1, 2, 0))
        img_yuv = rgb2yuv(img_data)
        y_channel = img_yuv[:, :, 0]
        uv_channels = torch.from_numpy(img_yuv[:, :, 1:3].transpose((2, 0, 1))).float()
        y_channel = torch.from_numpy(y_channel).unsqueeze(0).float()
        return y_channel, uv_channels
    


class ImageColorizer:

# rest same

    @staticmethod
    def _YUV_to_RGB(y_channel, uv_channels):
        yuv = torch.cat((y_channel, uv_channels), 0).numpy()
        yuv = yuv.transpose((1, 2, 0))
        return yuv2rgb(yuv.astype(np.float64))

    def _store_images(self, y_channel, uv_channels, path_dict, file_name):
        colored_image = self._YUV_to_RGB(y_channel, uv_channels)
        colored_image = np.clip(colored_image, 0, 1) 
        gray_image = y_channel.squeeze().numpy()
        plt.imsave(arr=gray_image, fname=f"{path_dict['grayscale']}{file_name}", cmap='gray')
        plt.imsave(arr=colored_image, fname=f"{path_dict['colorized']}{file_name}")

    def validation(self, loader, epoch_idx, save_images_flag, net, loss_function):
        net.eval()
        total_loss = 0
        images_saved = False
        for batch_idx, (y_input, uv_input) in enumerate(loader):
            y_input, uv_input = y_input.cuda(), uv_input.cuda()
            uv_output = net(y_input)
            loss = loss_function(uv_output, uv_input)
            total_loss += loss.item()
            if save_images_flag and not images_saved:
                images_saved = True
                paths = {
                    'grayscale': '/content/drive/MyDrive/585_project_f04/outputs/gray/',
                    'colorized': '/content/drive/MyDrive/585_project_f04/outputs/color/',
                    'ground_truth': '/content/drive/MyDrive/585_project_f04/outputs/ground_truth/'
                }
                name = f'img-{batch_idx * loader.batch_size + 0}-epoch-{epoch_idx + 1}.jpg'
                self._store_images(y_input[0].cpu(), uv_output[0].detach().cpu(), paths, name)


                gt_image = np.concatenate((y_input[0].cpu(), uv_input[0].cpu()), axis=0).transpose((1, 2, 0))
                gt_image = yuv2rgb(gt_image.astype(np.float64))
                gt_image = np.clip(gt_image, 0, 1)
                plt.imsave(arr=gt_image, fname=f"{paths['ground_truth']}{name}")

            self.writer.add_scalar("Validation Loss", loss.item(), epoch_idx * len(loader) + batch_idx)
        average_loss = total_loss / len(loader)
        self.val_loss_history.append(average_loss)
        print(f'Epoch: [{epoch_idx + 1}] Average validation Loss: {average_loss:.6f}')

# rest same