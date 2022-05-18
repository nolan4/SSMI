import time
import copy

from torch._C import *

from model.data_loader import *
from model.architecture import *
from model.unet_fcn import *


# https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision/model/net.py

num_class = 4
num_epochs = 10

batch_size = 4
num_workers = 1
epochs = 50     
learning_rate = 0.001

# scheduler param
step_size = .2
gamma = .9

dataset_path = '/Volumes/Seagate2/datasets/CAMUS/'

# dataset = SSEchoDataset(dataset_path, 'training', ImQ=['Poor','Medium','Good'], Chambers=['2','4'], SysDia=['ES','ED'], transform=transforms.Compose([ZeroPad((1500, 1100)), ToTensor()]))
# test_dataset = SSEchoDataset(dataset_path, 'testing', ImQ=['Poor','Medium','Good'], Chambers=['2','4'], SysDia=['ES','ED'], transform=transforms.Compose([ZeroPad((1500, 1100)), ToTensor()]))
dataset = SSEchoDataset(dataset_path, 'training', ImQ=['Poor','Medium','Good'], Chambers=['2','4'], SysDia=['ES','ED'])
test_dataset = SSEchoDataset(dataset_path, 'testing', ImQ=['Poor','Medium','Good'], Chambers=['2','4'], SysDia=['ES','ED'])

train_size = int(.9*len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

######## data is good to go at this point


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        # torch.nn.init.xavier_uniform(m.bias.data)        
        torch.nn.init.zeros_(m.bias.data)    


# initialize the model
# unet_model = UNet(num_class=num_class)
unet_model = FCN(num_class=num_class)
unet_model.apply(init_weights)

# check for GPU
use_gpu = torch.cuda.is_available()
if use_gpu:
    print('Using GPU...')
    unet_model = unet_model.cuda()

# define criterion/optimizer/scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(unet_model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


#######

def train(model=unet_model, criterion=criterion, optimizer=optimizer, scheduler=scheduler, num_epochs=num_epochs):

    # since = time.time()

    # best_model_wts = copy.deepcopy(model.state_dict())
    # best_acc = 0.0

    min_val_loss = np.finfo(np.float32()).max
    losses_list = []
    val_losses_list = []

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for iter, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            # print(data['scan'].size())
            X = data['scan']
            Y = data['gt']
            YM = data['gt_masks']

            print('scan', X.size())
            print('gt', Y.size())
            print('gt_masks', YM.size())

            # zero the parameter gradients
            optimizer.zero_grad()

            if use_gpu:
                inputs = X.cuda()
                gt = Y.cuda()
            else:
                inputs, gt = X, Y


            # forward + backward + optimize
            outputs = unet_model(inputs)
            
            print('outputs: \n')
            print(outputs.shape)
            
            print('gt: \n')
            print(gt.shape)

            loss = criterion(outputs, gt.long())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if iter % 100 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))

    print('Finished Training')

    # for epoch in range(epochs):

    #     ts = time.time()
    #     losses = 0

    #     for i, data in enumerate(train_loader, 0):
    #         inputs_, labels_ = data

    #         print()

    #         # print('inputs from train_loader:', inputs)
    #         # print('labels from train_loader:', labels)

    #        if iter % 100 == 0:
    #            print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))
        
    #     average_loss = losses / len(train_loader)
    #     losses_list.append(average_loss)
    #     print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
    #     print(f"Average Loss: {average_loss}")

    #     # early stopping
    #     val_loss, _, val_ious = val(epoch)
    #     val_losses_list.append(val_loss)
    #     if val_loss < min_val_loss:
    #         min_val_loss = val_loss
    #         torch.save(net, 'best_model')

    #     Unet.train()



# def val():


# def test():




if __name__ == '__main__':

    train()
    # val()
    # test()