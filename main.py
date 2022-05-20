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

# dataset = SSEchoDataset(dataset_path, 'training', ImQ=['Poor','Medium','Good'], Chambers=['2','4'], SysDia=['ES','ED'])
# test_dataset = SSEchoDataset(dataset_path, 'testing', ImQ=['Poor','Medium','Good'], Chambers=['2','4'], SysDia=['ES','ED'])
dataset = SSEchoDataset(dataset_path, 'training', ImQ=['Good'], Chambers=['4'], SysDia=['ES','ED'])
test_dataset = SSEchoDataset(dataset_path, 'testing', ImQ=['Good'], Chambers=['4'], SysDia=['ES','ED'])

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
        ts = time.time() # time how long it takes to iterate through train_loader
        running_loss = 0.0 # accumulated loss over each minibatch

        for iter, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            # print(data['scan'].size())
            X = data['scan']
            Y = data['gt']
            YM = data['gt_masks']

            # print('scan', X.size())
            # print('gt', Y.size())
            # print('gt_masks', YM.size())

            # zero the parameter gradients
            optimizer.zero_grad()

            if use_gpu:
                inputs = X.cuda()
                gt = Y.cuda()
            else:
                inputs, gt = X, Y


            # forward + backward + optimize
            outputs = unet_model(inputs)

            loss = criterion(outputs, gt.long())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if iter % 3 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))


        average_loss = running_loss / len(train_loader)
        losses_list.append(average_loss)
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        print(f"Average Loss: {average_loss}")


        # implement early stopping
        val_loss, val_ious = val(epoch) # after each epoch, perform validation
        val_losses_list.append(val_loss)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(unet_model, 'best_model')

        unet_model.train()

    plot_losses()


def plot_losses():

    train_loss_stds = np.zeros(epochs)
    train_loss_averages = np.zeros(epochs)

    val_loss_stds = np.zeros(epochs)
    val_loss_averages = np.zeros(epochs)

    for m in range(epochs):
        val_loss_averages[m] = torch.mean(epoch_val_loss[m])
        train_loss_averages[m] = torch.mean(epoch_train_loss[m])

    x = np.arange(epochs) + 1
    plt.clf()
    plt.plot(x, val_loss_averages, label='Validation dataset')
    plt.plot(x, train_loss_averages, label='Train dataset')
    
    print('val_loss_averages:', val_loss_averages)
    print('train_loss_averages:', train_loss_averages)

    plt.xlabel('Epoch')
    plt.ylabel('Normalized average cross-entropy loss')
    plt.legend(loc='best')
    filename = f"plots/{description}_loss.pdf"
    plt.savefig(filename)



def eval_model(dataloader, model):
    num_samples = len(dataloader)
    num_classes = 4

    losses = 0
    pixel_accuracies = 0
    ious = 0 
    five_ious = {0 : [0, 0], 2 : [0, 0], 9 : [0, 0], 17 : [0, 0], 25 : [0, 0]}
    unions = torch.zeros(1)
    intersections = torch.zeros(1)

    flattened_unions = torch.zeros(1)
    flattened_inters = torch.zeros(1)
    
    criterion1 = nn.CrossEntropyLoss()
    ts = time.time()
    for iter, (X, tar, Y) in enumerate(dataloader):
        if use_gpu:
            inputs = X.cuda() # Move your inputs onto the gpu
            labels = Y.cuda() # Move your labels onto the gpu
            targets = tar.cuda()
        else:
            inputs, labels = X, Y # Unpack variables into inputs and labels

        # print('inputs shape:', inputs.shape)
        outputs = model(inputs) # N x C x H x W
        # print('outputs[0]:', torch.sum(outputs[0]))
        # print('outputs shape:', outputs.shape)
        # print('labels shape:', labels.shape)
        loss = criterion1(outputs, labels.long())
        #loss = criterion(outputs, targets)

        if iter % 100 == 0:
            print("eval_model iter{}, loss: {}".format(iter, loss.item()))
        
        losses += loss
        flattened_outputs = torch.argmax(outputs, dim=1) # along class dim
        if iter == 0:
            generate_segmented_img(flattened_outputs, 'predicted')
            generate_segmented_img(labels, 'actual')
        pixel_accuracies += pixel_acc(flattened_outputs, labels)
        # ious += torch.nansum(iou(outputs, targets))
        epoch_inters, epoch_unions = iou(outputs, targets)

        # Update 5 ious
        for key in five_ious.keys():
            five_ious[key][0] += epoch_inters[key]
            five_ious[key][1] += epoch_unions[key]
        
        intersections[0] += torch.nansum(epoch_inters)
        unions[0] += torch.nansum(epoch_unions)

    print("Finish eval_model, time elapsed {}".format(time.time() - ts))
    avg_loss = losses / num_samples
    avg_pix_acc = pixel_accuracies / num_samples
    # avg_ious = ious / (num_samples * num_classes)
    avg_ious = intersections[0] / unions[0]

    # Compute 5 ious
    for key in five_ious.keys():
        inter = five_ious[key][0]
        union = five_ious[key][1]
        print("{} iou: {}".format(label_mappings[key][0], inter / union))


    return avg_loss, avg_pix_acc, avg_ious



"""
Given a batch of labeled images (size N x H x W), 
generate the first segmented colored image
"""
def generate_segmented_img(images, filename):
    imgs = images.cpu()
    height, width = imgs[0].shape
    img = np.asarray(imgs[0]).flatten()
    # for img in imgs:
    # seg_img = [label_mappings[pix][2] for pix in img]
    seg_img = [[label_mappings[pix][2][0], \
                label_mappings[pix][2][1], \
                label_mappings[pix][2][2]] \
                for pix in img]
    seg_img = np.asarray(seg_img).reshape(height, width, 3).astype(np.float32)
    plt.imsave(f"{filename}.pdf", seg_img)
    # plt.show()  


def val(epoch):
    val_model = nn.Sequential( unet_model, nn.Softmax(dim=1) )
    val_model.eval() # Put in eval mode to evaluate!
    #Complete this function - Calculate loss, accuracy and IoU for every epoch
    # Make sure to include a softmax after the output from your model
    print('-------- Evaluating validation set --------')
    with torch.no_grad():
        avg_val_loss, avg_pix_acc, avg_ious = eval_model(val_loader, val_model)
    
    print('val at epoch{}: {} loss, {} pix acc, {} iou'.format(epoch, avg_val_loss, avg_pix_acc, avg_ious))
    return avg_val_loss, avg_pix_acc, avg_ious
    
    
def test():
    test_model = torch.load("best_model")
    test_model = nn.Sequential( test_model, nn.Softmax(dim=1) )
    test_model.eval()
    #Complete this function - Calculate accuracy and IoU 
    # Make sure to include a softmax after the output from your model
    print('-------- Evaluating test set --------')
    with torch.no_grad():
        _, avg_pix_acc, avg_ious = eval_model(test_loader, test_model)

    print('test: {} pix acc, {} iou'.format(avg_pix_acc, avg_ious))
    return avg_pix_acc, avg_ious










if __name__ == '__main__':

    train()
    # val()
    # test()