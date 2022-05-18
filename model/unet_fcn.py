import torch.nn as nn
import torch
import torch.nn.functional as F


class FCN(nn.Module):

    def __init__(self, num_class):
        super().__init__()
        self.num_class = num_class
        self.relu       = nn.ReLU(inplace=True)
        self.maxpool    = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)

        # 3, 64 / 64, 64
        # self.conv1      = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0, dilation=1) # implies no zero-padding
        self.conv1      = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=0, dilation=1) # implies no zero-padding
        self.bnd1       = nn.BatchNorm2d(64)
        self.conv2      = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, dilation=1)
        self.bnd2       = nn.BatchNorm2d(64)

        # 64, 128 / 128, 128
        self.conv3      = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0, dilation=1)
        self.bnd3       = nn.BatchNorm2d(128)
        self.conv4      = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0, dilation=1)
        self.bnd4       = nn.BatchNorm2d(128)

        # 128, 256 / 256, 256
        self.conv5      = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0, dilation=1)
        self.bnd5       = nn.BatchNorm2d(256)
        self.conv6      = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0, dilation=1)
        self.bnd6       = nn.BatchNorm2d(256)

        # 256, 512 / 512, 512
        self.conv7      = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0, dilation=1)
        self.bnd7       = nn.BatchNorm2d(512)
        self.conv8      = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0, dilation=1)
        self.bnd8       = nn.BatchNorm2d(512)

        # # 512, 1024 / 1024, 1024
        # self.conv9      = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=0, dilation=1)
        # self.bnd9       = nn.BatchNorm2d(1024)
        # self.conv10     = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=0, dilation=1)
        # self.bnd10      = nn.BatchNorm2d(1024)

        # # 1024, 512
        # self.deconv1    = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=1, padding=0, dilation=1)
        # self.bnd_d1     = nn.BatchNorm2d(512)

        # # 128, 256 / 256, 256
        # self.conv11     = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=0, dilation=1)
        # self.bnd11      = nn.BatchNorm2d(512)
        # self.conv12     = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0, dilation=1)
        # self.bnd12      = nn.BatchNorm2d(512)

        # 512, 256
        self.deconv2    = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=1, padding=0, dilation=1) # combine this with prev output
        self.bnd_d2     = nn.BatchNorm2d(256)

        # 512, 256 / 256, 256
        self.conv13     = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0, dilation=1)
        self.bnd13      = nn.BatchNorm2d(256)
        self.conv14     = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0, dilation=1)
        self.bnd14      = nn.BatchNorm2d(256)

        # 256, 128
        self.deconv3    = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=1, padding=0, dilation=1) # combine this with prev output
        self.bnd_d3     = nn.BatchNorm2d(128)

        # 256, 128 / 128, 128
        self.conv15     = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0, dilation=1)
        self.bnd15      = nn.BatchNorm2d(128)
        self.conv16     = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0, dilation=1)
        self.bnd16      = nn.BatchNorm2d(128)

        # 128, 64
        self.deconv4    = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=1, padding=0, dilation=1) # combine this with prev output
        self.bnd_d4     = nn.BatchNorm2d(64)

        # 128, 64 / 64, 64
        self.conv17     = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0, dilation=1)
        self.bnd17      = nn.BatchNorm2d(64)
        self.conv18     = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, dilation=1)
        self.bnd18      = nn.BatchNorm2d(64)

        # 64, 27 
        self.conv19     = nn.Conv2d(64, 4, kernel_size=1, stride=1, padding=4, dilation=1)
        self.bnd19      = nn.BatchNorm2d(4)

    def forward(self, x):

        # print('A')
        x1 = self.bnd1(self.relu(self.conv1(x)))
        x2 = self.bnd2(self.relu(self.conv2(x1))) # U1A
        x3 = self.maxpool(x2)
        # 1st row in fig 1 ^
        # print('x2', x2.shape)

        # print('B')
        x4 = self.bnd3(self.relu(self.conv3(x3)))
        x5 = self.bnd4(self.relu(self.conv4(x4))) # U2A
        x6 = self.maxpool(x5)
        # 2nd row in fig 1 ^
        # print('x6', x6.shape)

        # print('C')
        x7 = self.bnd5(self.relu(self.conv5(x6)))
        x8 = self.bnd6(self.relu(self.conv6(x7))) # U3A
        x9 = self.maxpool(x8)
        # 3rd row in fig 1 ^
        # print('x9', x9.shape)

        # print('D')
        x10 = self.bnd7(self.relu(self.conv7(x9)))
        x11 = self.bnd8(self.relu(self.conv8(x10))) # U4A
        x12 = self.bnd_d2(self.deconv2(x11)) # U3B
        # x12 = self.maxpool(x11)
        # 4th row in fig 1 ^
        # print('x12', x12.shape)

        # x13 = self.bnd9(self.relu(self.conv9(x12)))
        # x14 = self.bnd10(self.relu(self.conv10(x13)))
        # x15 = self.bnd_d1(self.deconv1(x14)) # U4B
        # # BOTTOM OF U ^

        # x15 = nn.functional.pad(input=x15, pad=(15,0,0,15), mode='constant')
        # # print(x15.shape, x11.shape)        
        # # a = torch.cat((x11,x15), dim=1)
        # x16 = self.bnd11(self.relu(self.conv11(torch.cat((x11,x15), dim=1)))) # a is a combination of x15 and x11
        # x17 = self.bnd12(self.relu(self.conv12(x16)))
        # x18 = self.bnd_d2(self.deconv2(x17)) # U3B
        # # 4th row in fig 1 ^

        # x18 = nn.functional.pad(input=x18, pad=(36,0,0,36), mode='constant')
        x18 = nn.functional.pad(input=x12, pad=(137,0,0,187), mode='constant')
        print('x18', x18.shape, 'x8', x8.shape)        
        b = torch.cat((x8,x18), dim=1)
        x19 = self.bnd13(self.relu(self.conv13(b))) # b is a combination of x18 and x8
        x20 = self.bnd14(self.relu(self.conv14(x19)))
        x21 = self.bnd_d3(self.deconv3(x20)) # U2B
        # 3rd row in fig 1 ^
        print('x21', x21.shape)

        x21 = F.pad(input=x21, pad=(279,0,0,379), mode='constant')
        print('x21', x21.shape, 'x5', x5.shape)        
        c = torch.cat((x5,x21), dim=1)
        x22 = self.bnd15(self.relu(self.conv15(c))) # c is a combination of x21 and x5
        x23 = self.bnd16(self.relu(self.conv16(x22)))
        x24 = self.bnd_d4(self.deconv4(x23)) #U1B
        # 2nd row in fig 1 ^
        print('x24', x24.shape)

        x24 = nn.functional.pad(input=x24, pad=(555,0,0,755), mode='constant')
        print('x24', x24.shape, 'x2', x2.shape)        
        d = torch.cat((x2,x24), dim=1)
        x25 = self.bnd17(self.relu(self.conv17(d))) # d is a combination of x24 and x2
        x26 = self.bnd18(self.relu(self.conv18(x25)))
        x27 = self.bnd19(self.conv19(x26))
        # 1st row in fig 1 ^

        score = x27



        return score  # size=(N, n_class, x.H/1, x.W/1)


