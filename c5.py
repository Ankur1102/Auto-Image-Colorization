import torch.nn as nn
import torch
# uses unet architecture to crerate ab channels
class ColorizationNet5(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super(ColorizationNet5, self).__init__()
        
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        def upconv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
                nn.ReLU(inplace=True)
            )

        self.encoder = nn.Sequential(
            conv_block(1, 64),
            nn.MaxPool2d(2, 2),
            conv_block(64, 128),
            nn.MaxPool2d(2, 2),
            conv_block(128, 256),
            nn.MaxPool2d(2, 2),
            conv_block(256, 512),
            nn.MaxPool2d(2, 2),
            conv_block(512, 1024),
        )

        self.upconv1 = upconv_block(1024, 512)
        self.decoder1 = conv_block(1024, 512)
        self.upconv2 = upconv_block(512, 256)
        self.decoder2 = conv_block(512, 256)
        self.upconv3 = upconv_block(256, 128)
        self.decoder3 = conv_block(256, 128)
        self.upconv4 = upconv_block(128, 64)
        self.decoder4 = conv_block(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        enc1 = self.encoder[0](x)
        enc2 = self.encoder[2](self.encoder[1](enc1))
        enc3 = self.encoder[4](self.encoder[3](enc2))
        enc4 = self.encoder[6](self.encoder[5](enc3))
        enc5 = self.encoder[8](self.encoder[7](enc4))

        dec1 = self.upconv1(enc5)
        dec1 = torch.cat([enc4, dec1], dim=1)
        dec1 = self.decoder1(dec1)

        dec2 = self.upconv2(dec1)
        dec2 = torch.cat([enc3, dec2], dim=1)
        dec2 = self.decoder2(dec2)

        dec3 = self.upconv3(dec2)
        dec3 = torch.cat([enc2, dec3], dim=1)
        dec3 = self.decoder3(dec3)

        dec4 = self.upconv4(dec3)
        dec4 = torch.cat([enc1, dec4], dim=1)
        dec4 = self.decoder4(dec4)

        out = self.final_conv(dec4)
        return out