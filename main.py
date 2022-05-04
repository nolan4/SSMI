import time
import copy

from model.data_loader import *
from model.architecture import *


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

dataset_path = 'Datasets/'

dataset = SSEchoDataset(dataset_path, 'training', ImQ=['Poor','Medium','Good'], Chambers=['2','4'], SysDia=['ES','ED'], transform=transforms.Compose([ZeroPad((1200, 800)), ToTensor()]))
test_dataset = SSEchoDataset(dataset_path, 'testing', ImQ=['Poor','Medium','Good'], Chambers=['2','4'], SysDia=['ES','ED'], transform=transforms.Compose([ZeroPad((1200, 800)), ToTensor()]))

train_size = int(.9*len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

######## data is good to go at this point


# use xavier weight initialization
# def init_weights(m):
#     if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
#         nn.init.xavier_uniform_(m.weight.data)
#         nn.init_zeros(m.bias.data)


# initialize the model
unet_model = UNet(num_class=num_class)
# unet_model.apply(init_weights)

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

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):

        ts = time.time()
        losses = 0

        for i, data  in enumerate(train_loader, 0):

        
            # zero the parameter gradients
            optimizer.zero_grad()

            if use_gpu:
                inputs = data['scan'].cuda()
                labels = data['gt'].cuda()
            else:
                inputs, labels = data['scan'], data['gt']


            #zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = UNet(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            losses += loss

            if iter % 2000 == 1:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))
                losses = 0.0
        
        average_loss = losses / len(train_loader)
        losses_list.append(avergae_loss)
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        print(f"Average Loss: {average_loss}")

        # early stopping
        val_loss, _, val_ious = val(epoch)
        val_losses_list.append(val_loss)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(net, 'best_model')

        Unet.train()



#def val():


#def test():




if __name__ == '__main__':

    train()
    # val()
    # test()