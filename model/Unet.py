import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):

    # maybe include batch normalization??

    #### define model
    def __init__(self):
        super().__init__()

# torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
# torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
# torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)
# torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)

        self.convA1 = nn.Conv2d(1, 64, 3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        self.convA2 = nn.Conv2d(64, 64, 3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

        self.poolA1 = nn.MaxPool2d(2)

        self.convB1 = nn.Conv2d(64, 128, 3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        self.convB2 = nn.Conv2d(128, 128, 3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        
        self.poolB1 = nn.MaxPool2d(2)

        self.convC1 = nn.conv2d(128, 256, 3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        self.convC2 = nn.conv2d(256, 256, 3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

        self.poolC1 = nn.MaxPool2d(2)

        self.convD1 = nn.conv2d(128, 512, 3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        self.convD2 = nn.conv2d(512, 512, 3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

        self.poolD1 = nn.MaxPool2d(2)

        self.convE1 = nn.Conv2d(512, 1024, 3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        self.convE2 = nn.Conv2d(1024, 1024, 3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

        self.upconvD1 = nn.ConvTranspose2d(1024, 512, 2, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)

        self.convD3 = nn.conv2d(1024, 512, 3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        self.convD4 = nn.conv2d(512, 512, 3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

        self.upconvC1 = nn.ConvTranspose2d(512, 256, 2, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)

        self.convC3 = nn.conv2d(512, 256, 3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        self.convC4 = nn.conv2d(256, 256, 3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

        self.upconvB1 = nn.ConvTranspose2d(256, 128, 2, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)

        self.convB3 = nn.conv2d(256, 128, 3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        self.convB4 = nn.conv2d(128, 128, 3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

        self.upconvA1 = nn.ConvTranspose2d(128, 64, 2, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)

        self.convA3 = nn.conv2d(128, 64, 3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        self.convA4 = nn.conv2d(64, 64, 3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

        self.linear = nn.Linear(123456789, 4) # 4 classes in a segmented image? (will need to change 123456789)


        #### define forward pass
    def forward(self, input):

        x1 = F.relu(self.convA1(input))
        x2 = F.relu(self.convA2(x1))
        x3 = self.poolA1(x2)

        x4 = F.relu(self.convB1(x3))
        x5 = F.relu(self.convB2(x4))
        x6 = self.poolB1(x5)

        x7 = F.relu(self.convC1(x6))
        x8 = F.relu(self.convC2(x7))
        x9 = self.poolC1(x8)

        x10 = F.relu(self.convD1(x3))
        x11 = F.relu(self.convD2(x4))
        x12 = self.poolD1(x5)

        # now at bottom of U
        # torch.cat(tensors, dim=0, *, out=None)
        # (will need to fix concatination dimensions in torch.cat)


        x13 = F.relu(self.convD3(x12))
        x14 = F.relu(self.convD4(x13))
        x15 = self.upconvD1(x14)

        x15x11 = torch.cat(x15, torch.copy(x11))
        x16 = F.relu(self.convC3(x15x11))
        x17 = F.relu(self.convC4(x16))
        x18 = self.upconvC1(x17)

        x18x8 = torch.cat(x18, torch.copy(x8))
        x19 = F.relu(self.convB3(x18x8))
        x20 = F.relu(self.convB4(x19))
        x21 = self.upconvB1(x20)

        x21x5 = torch.cat(x21, torch.copy(x5))
        x22 = F.relu(self.convA3(x21x5))
        x23 = F.relu(self.convA4(x22))
        x24 = self.upconvA1(x23)

        x24x2 = torch.cat(x24, torch.copy(x2))
        x25 = F.relu(self.convA3(x24x2))
        x26 = F.relu(self.convA4(x25))
        x27 = self.upconvA1(x26)

        output = self.linear(x27)

        return output



    def loss(outputs, labels):


        return


    def accuracy(outputs, labels):

        return


    metrics = {
        'accuracy': accuracy,
        # add more if desired
             }






    