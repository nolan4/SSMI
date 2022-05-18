import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils

class UNet(nn.Module):

    # maybe include batch normalization??

    #### define model
    def __init__(self, num_class):
        super().__init__()

# torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
# torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
# torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)
# torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
# torch.nn.functional.interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False)


        self.convA1 = nn.Conv2d(1, 64, 3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.convA2 = nn.Conv2d(64, 64, 3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.poolA1 = nn.MaxPool2d(2)

        self.convB1 = nn.Conv2d(64, 128, 3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.convB2 = nn.Conv2d(128, 128, 3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.poolB1 = nn.MaxPool2d(2)

        self.convC1 = nn.Conv2d(128, 256, 3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.convC2 = nn.Conv2d(256, 256, 3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.poolC1 = nn.MaxPool2d(2)

        self.convD1 = nn.Conv2d(256, 512, 3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.convD2 = nn.Conv2d(512, 512, 3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.poolD1 = nn.MaxPool2d(2)

        self.convE1 = nn.Conv2d(512, 1024, 3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.convE2 = nn.Conv2d(1024, 1024, 3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.upconvE1 = nn.ConvTranspose2d(1024, 512, 2, stride=1, padding=1, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')


        self.convD3 = nn.Conv2d(1024, 512, 3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.convD4 = nn.Conv2d(512, 512, 3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.upconvD1 = nn.ConvTranspose2d(512, 256, 2, stride=1, padding=1, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')


        self.convC3 = nn.Conv2d(512, 256, 3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.convC4 = nn.Conv2d(256, 256, 3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.upconvC1 = nn.ConvTranspose2d(256, 128, 2, stride=1, padding=1, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')


        self.convB3 = nn.Conv2d(256, 128, 3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.convB4 = nn.Conv2d(128, 128, 3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.upconvB1 = nn.ConvTranspose2d(128, 64, 2, stride=1, padding=1, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')

        # self.upconvA1 = nn.ConvTranspose2d(128, 64, 2, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')

        self.convA3 = nn.Conv2d(128, 64, 3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.convA4 = nn.Conv2d(64, 64, 3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.convA5 = nn.Conv2d(64, 4, 1, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros') # 4 classes in a segmented image? (will need to change 123456789)



    # def tensorCenterCrop(self, tensor_big, tensor_small):
    #     _, _, H, W = tensor_small.shape
    #     out_tensor = transforms.CenterCrop([H, W])(tensor_big)
    #     # print(out_tensor.size(), tensor_small.size())
    #     return out_tensor


        #### define forward pass
    def forward(self, input):

        # input = input.unsqueeze(1)
        print('input size:', input.size())
        print('input type:', type(input))
        print('input data type:', input.dtype)

        x1 = F.relu(self.convA1(input))
        x2 = F.relu(self.convA2(x1))
        x3 = self.poolA1(x2)

        # print('x1', x1.size())
        # print('x2', x2.size())
        print('x3', x3.size())

        x4 = F.relu(self.convB1(x3))
        x5 = F.relu(self.convB2(x4))
        x6 = self.poolB1(x5)

        # print('x4', x4.size())
        # print('x5', x5.size())
        print('x6', x6.size())

        x7 = F.relu(self.convC1(x6))
        x8 = F.relu(self.convC2(x7))
        x9 = self.poolC1(x8)

        # print('x7', x4.size())
        # print('x8', x5.size())
        print('x9', x6.size())

        x10 = F.relu(self.convD1(x9))
        x11 = F.relu(self.convD2(x10))
        x12 = self.poolD1(x11)

        # print('x10', x10.size())
        # print('x11', x11.size())
        print('x12', x12.size())

        # now at bottom of U
        #######################

        x13 = F.relu(self.convE1(x12))
        x14 = F.relu(self.convE2(x13))
        x15 = self.upconvE1(x14)

        # print('x13', x13.size())
        # print('x14', x14.size())
        print('x15', x15.size())

        #######################

        # torch.cat(tensors, dim=0, *, out=None)
        # F.interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False)

        x15x11_interped = torch.cat((x11, F.interpolate(x15, x11.size()[2:])), 1)
        x16 = F.relu(self.convD3(x15x11_interped))
        x17 = F.relu(self.convD4(x16))
        x18 = self.upconvD1(x17)

        # print('x15x11_interped', x15x11_interped.size())
        # print('x16', x16.size())
        # print('x17', x17.size())
        print('x18', x18.size())

        x18x8_interped = torch.cat((x8, F.interpolate(x18, x8.size()[2:])), 1)
        x19 = F.relu(self.convC3(x18x8_interped))
        x20 = F.relu(self.convC4(x19))
        x21 = self.upconvC1(x20)

        # print('x18x8_interped', x18x8_interped.size())
        # print('x19', x19.size())
        # print('x20', x20.size())
        print('x21', x21.size())

        x21x5_interped = torch.cat((x5, F.interpolate(x21, x5.size()[2:])), 1)
        x22 = F.relu(self.convB3(x21x5_interped))
        x23 = F.relu(self.convB4(x22))
        x24 = self.upconvB1(x23)

        # print('x21x5_interped', x21x5_interped.size())
        # print('x22', x22.size())
        # print('x23', x23.size())
        print('x24', x24.size())

        x24x2_interped = torch.cat((x2, F.interpolate(x24, x2.size()[2:])), 1)
        x25 = F.relu(self.convA3(x24x2_interped))
        x26 = F.relu(self.convA4(x25))
        outputs = self.convA5(x26)

        # print('x24x2_interped', x24x2_interped.size())
        # print('x25', x25.size())
        # print('x26', x26.size())
        print('outputs', outputs.size())

        outputs_final = F.interpolate(outputs, input.size()[2:])
        print('outputs_final', outputs_final.size())

        return outputs_final



    # def loss(outputs, labels):


    #     return


    # def accuracy(outputs, labels):

    #     return


    # metrics = {
    #     'accuracy': accuracy,
    #     # add more if desired
    #          }






    