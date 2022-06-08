import torch.nn as nn
import torch
import torch.nn.functional as F


class FCN(nn.Module):

    def __init__(self, num_class):
        super().__init__()

        self.relu       = nn.ReLU(inplace=True)
        self.maxpool    = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)

        # 3, 64 / 64, 64
        self.conv1      = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=0, dilation=1) # implies no zero-padding
        self.bn1       = nn.BatchNorm2d(64)
        self.conv2      = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, dilation=1)
        self.bn2       = nn.BatchNorm2d(64)

        # 64, 128 / 128, 128
        self.conv3      = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0, dilation=1)
        self.bn3       = nn.BatchNorm2d(128)
        self.conv4      = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0, dilation=1)
        self.bn4       = nn.BatchNorm2d(128)

        # 128, 256 / 256, 256
        self.conv5      = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0, dilation=1)
        self.bn5       = nn.BatchNorm2d(256)
        self.conv6      = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0, dilation=1)
        self.bn6       = nn.BatchNorm2d(256)

        # 256, 512 / 512, 512
        self.conv7      = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0, dilation=1)
        self.bn7       = nn.BatchNorm2d(512)
        self.conv8      = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0, dilation=1)
        self.bn8       = nn.BatchNorm2d(512)

        # 512, 1024 / 1024, 1024
        self.conv9      = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=0, dilation=1)
        self.bn9       = nn.BatchNorm2d(1024)
        self.conv10     = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=0, dilation=1)
        self.bn10      = nn.BatchNorm2d(1024)

        # 1024, 512
        self.deconv1    = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=1, padding=0, dilation=1)
        self.bn_d1     = nn.BatchNorm2d(512)

        # 128, 256 / 256, 256
        self.conv11     = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=0, dilation=1)
        self.bn11      = nn.BatchNorm2d(512)
        self.conv12     = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0, dilation=1)
        self.bn12      = nn.BatchNorm2d(512)

        # 512, 256
        self.deconv2    = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=1, padding=0, dilation=1) # combine this with prev output
        self.bn_d2     = nn.BatchNorm2d(256)

        # 512, 256 / 256, 256
        self.conv13     = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0, dilation=1)
        self.bn13      = nn.BatchNorm2d(256)
        self.conv14     = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0, dilation=1)
        self.bn14      = nn.BatchNorm2d(256)

        # 256, 128
        self.deconv3    = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=1, padding=0, dilation=1) # combine this with prev output
        self.bn_d3     = nn.BatchNorm2d(128)

        # 256, 128 / 128, 128
        self.conv15     = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0, dilation=1)
        self.bn15      = nn.BatchNorm2d(128)
        self.conv16     = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0, dilation=1)
        self.bn16      = nn.BatchNorm2d(128)

        # 128, 64
        self.deconv4    = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=1, padding=0, dilation=1) # combine this with prev output
        self.bn_d4     = nn.BatchNorm2d(64)

        # 128, 64 / 64, 64
        self.conv17     = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0, dilation=1)
        self.bn17      = nn.BatchNorm2d(64)
        self.conv18     = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, dilation=1)
        self.bn18      = nn.BatchNorm2d(64)

        # 64, 27 
        self.conv19     = nn.Conv2d(64, num_class, kernel_size=1, stride=1, padding=4, dilation=1)
        self.bn19      = nn.BatchNorm2d(num_class)

    def forward(self, x):
        
        # NO BATCH NORM.
        # padding is used instead of cropping and tiling to deal with lost border pixels

        # row 1 - down
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1)) # U1A
        x3 = self.maxpool(x2)

        # row 2 - down
        x4 = self.relu(self.conv3(x3))
        x5 = self.relu(self.conv4(x4)) # U2A
        x6 = self.maxpool(x5)

        # row 3 - down
        x7 = self.relu(self.conv5(x6))
        x8 = self.relu(self.conv6(x7)) # U3A
        x9 = self.maxpool(x8)

        # row 4 - down
        x10 = self.relu(self.conv7(x9))
        x11 = self.relu(self.conv8(x10)) # U4A
        x12 = self.maxpool(x11)

        # row 5 - up (BOTTOM OF U)
        x13 = self.relu(self.conv9(x12))
        x14 = self.relu(self.conv10(x13))
        x15 = self.deconv1(x14) # U4B

        # row 4 - up
        x15 = F.pad(input=x15, pad=(x11.size(dim=3)-x15.size(dim=3),0,0,x11.size(dim=2)-x15.size(dim=2)), mode='constant')
        # print('x15', x15.shape, 'x11', x11.shape)        
        a = torch.cat((x11,x15), dim=1)
        x16 = self.relu(self.conv11(a)) # a is a combination of x15 and x11
        x17 = self.relu(self.conv12(x16))
        x18 = self.deconv2(x17) # U3B

        # row 3 - up
        x18 = nn.functional.pad(input=x18, pad=(x8.size(dim=3)-x18.size(dim=3),0,0,x8.size(dim=2)-x18.size(dim=2)), mode='constant')
        # print('x18', x18.shape, 'x8', x8.shape)        
        b = torch.cat((x8,x18), dim=1)
        x19 = self.relu(self.conv13(b)) # b is a combination of x18 and x8
        x20 = self.relu(self.conv14(x19))
        x21 = self.deconv3(x20) # U2B

        # row 2 - up
        x21 = F.pad(input=x21, pad=(x5.size(dim=3)-x21.size(dim=3),0,0,x5.size(dim=2)-x21.size(dim=2)), mode='constant')
        # print('x21', x21.shape, 'x5', x5.shape)        
        c = torch.cat((x5,x21), dim=1)
        x22 = self.relu(self.conv15(c)) # c is a combination of x21 and x5
        x23 = self.relu(self.conv16(x22))
        x24 = self.deconv4(x23) #U1B

        # row 1 - out
        x24 = F.pad(input=x24, pad=(x2.size(dim=3)-x24.size(dim=3),0,0,x2.size(dim=2)-x24.size(dim=2)), mode='constant')
        # print('x24', x24.shape, 'x2', x2.shape)        
        d = torch.cat((x2,x24), dim=1)
        x25 = self.relu(self.conv17(d)) # d is a combination of x24 and x2
        x26 = self.relu(self.conv18(x25))
        score = self.conv19(x26)

#         # padding is used instead of cropping and tiling to deal with lost border pixels

#           WITH BATCH NORM               
           
#         # row 1 - down
#         x1 = self.bn1(self.relu(self.conv1(x)))
#         x2 = self.bn2(self.relu(self.conv2(x1))) # U1A
#         x3 = self.maxpool(x2)

#         # row 2 - down
#         x4 = self.bn3(self.relu(self.conv3(x3)))
#         x5 = self.bn4(self.relu(self.conv4(x4))) # U2A
#         x6 = self.maxpool(x5)

#         # row 3 - down
#         x7 = self.bn5(self.relu(self.conv5(x6)))
#         x8 = self.bn6(self.relu(self.conv6(x7))) # U3A
#         x9 = self.maxpool(x8)

#         # row 4 - down
#         x10 = self.bn7(self.relu(self.conv7(x9)))
#         x11 = self.bn8(self.relu(self.conv8(x10))) # U4A
#         x12 = self.maxpool(x11)

#         # row 5 - up (BOTTOM OF U)
#         x13 = self.bn9(self.relu(self.conv9(x12)))
#         x14 = self.bn10(self.relu(self.conv10(x13)))
#         x15 = self.bn_d1(self.deconv1(x14)) # U4B

#         # row 4 - up
#         x15 = F.pad(input=x15, pad=(x11.size(dim=3)-x15.size(dim=3),0,0,x11.size(dim=2)-x15.size(dim=2)), mode='constant')
#         # print('x15', x15.shape, 'x11', x11.shape)        
#         a = torch.cat((x11,x15), dim=1)
#         x16 = self.bn11(self.relu(self.conv11(a))) # a is a combination of x15 and x11
#         x17 = self.bn12(self.relu(self.conv12(x16)))
#         x18 = self.bn_d2(self.deconv2(x17)) # U3B

#         # row 3 - up
#         x18 = nn.functional.pad(input=x18, pad=(x8.size(dim=3)-x18.size(dim=3),0,0,x8.size(dim=2)-x18.size(dim=2)), mode='constant')
#         # print('x18', x18.shape, 'x8', x8.shape)        
#         b = torch.cat((x8,x18), dim=1)
#         x19 = self.bn13(self.relu(self.conv13(b))) # b is a combination of x18 and x8
#         x20 = self.bn14(self.relu(self.conv14(x19)))
#         x21 = self.bn_d3(self.deconv3(x20)) # U2B

#         # row 2 - up
#         x21 = F.pad(input=x21, pad=(x5.size(dim=3)-x21.size(dim=3),0,0,x5.size(dim=2)-x21.size(dim=2)), mode='constant')
#         # print('x21', x21.shape, 'x5', x5.shape)        
#         c = torch.cat((x5,x21), dim=1)
#         x22 = self.bn15(self.relu(self.conv15(c))) # c is a combination of x21 and x5
#         x23 = self.bn16(self.relu(self.conv16(x22)))
#         x24 = self.bn_d4(self.deconv4(x23)) #U1B

#         # row 1 - out
#         x24 = F.pad(input=x24, pad=(x2.size(dim=3)-x24.size(dim=3),0,0,x2.size(dim=2)-x24.size(dim=2)), mode='constant')
#         # print('x24', x24.shape, 'x2', x2.shape)        
#         d = torch.cat((x2,x24), dim=1)
#         x25 = self.bn17(self.relu(self.conv17(d))) # d is a combination of x24 and x2
#         x26 = self.bn18(self.relu(self.conv18(x25)))
#         score = self.bn19(self.conv19(x26))

        return score  # size=(batch_size, n_class, x.size[0], x.size[1])


