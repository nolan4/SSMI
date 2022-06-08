import time
import copy

from torch._C import *
from model.data_loader import *
from model.unet_fcn import *
from model.dropouttest import *
from model.shallow_unet import *
from utils import *


# https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision/model/net.py

batch_size = 10
num_workers = 0
num_epochs = 30
learning_rate = 0.01

# batch_size = 2
# num_workers = 0
# num_epochs = 2
# learning_rate = 0.01

# scheduler param
step_size = 1
gamma = 1

# sanity check
num_class = 4

# dataset_path = '/Volumes/Seagate2/datasets/CAMUS/'
dataset_path = '/home/nardolino/CAMUS/'

dataset = SSEchoDataset(dataset_path, 'training', ImQ=['Poor', 'Medium', 'Good'], Chambers=['2','4'], SysDia=['ES','ED'])
# test_dataset = SSEchoDataset(dataset_path, 'testing', ImQ=['Good'], Chambers=['4'], SysDia=['ES','ED'])
# demoEF_dataset = SSEchoDataset(dataset_path, 'demoEF', ImQ=['Good'], Chambers=['4'], SysDia=['ES','ED'])

train_size = int(.9*len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# you need to count the number of characters in your path up until /patient0xxx/
# for me, it is '/home/nardolino/CAMUS/training/patient0279/patient0279_4CH_ED_gt.mhd'
val_data_paths = [dataset.data_paths[i] for i in val_dataset.indices]

# calculate EF using these patients
val_patients = sorted(list(dict.fromkeys([path[0][38:42] for path in val_data_paths])), reverse=True)
# print(val_patients)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
# demoEF_loader = torch.utils.data.DataLoader(demoEF_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# print(demoEF_loader)

######## data is good to go at this point

# initialize the weights using xavier weight initialization
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        # torch.nn.init.xavier_uniform(m.bias.data)        
        torch.nn.init.zeros_(m.bias.data)    

# define 
def diceCoeffv2(pred, gt, eps=1e-5):
    pred = nn.Softmax2d()(pred)
  
    N = gt.size(0)
    pred_flat = pred.view(N,-1).cuda()
    gt_flat = gt.view(N,-1).cuda()

    tp = torch.sum(pred_flat * gt_flat, dim=1)
    fp = torch.sum(pred_flat, dim=1) - tp
    fn = torch.sum(gt_flat, dim=1) - tp
    loss = (2*tp + eps) / (2*tp + fp + fn + eps)

    return loss.sum() / N

class DiceLoss(nn.Module):
    __name__ = 'dice_loss'

    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, y_pr, y_gt):
        return 1 - diceCoeffv2(y_pr, y_gt)


# initialize the model
# unet_model = UNet(num_class=num_class)
# unet_model = FCN(num_class=num_class)
# unet_model = dropU(num_class=num_class)
unet_model = shallowDropU(num_class=num_class)
unet_model.apply(init_weights)

# check for GPU
use_gpu = torch.cuda.is_available()
if use_gpu:
    print('Using GPU...')
    unet_model = unet_model.cuda()

# define criterion/optimizer/scheduler
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(unet_model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


#######

def train(model=unet_model, criterion=criterion, optimizer=optimizer, scheduler=scheduler, num_epochs=num_epochs):

    # since = time.time()

    # best_model_wts = copy.deepcopy(model.state_dict())
    # best_acc = 0.0

    min_val_loss = np.finfo(np.float32()).max
    losses_list = []
    val_losses_list = []

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        ts = time.time() # time how long it takes to iterate through train_loader
        running_loss = 0.0 # accumulated loss over each minibatch

        for iter, data in enumerate(train_loader):
            # if iter>=5:
            #     continue
            # get the inputs; data is a list of [inputs, labels]
            X = data['scan']
            Y = data['gt']
            YM = data['gt_masks']

            # zero the parameter gradients
            optimizer.zero_grad()

            # check for GPU speedup
            if torch.cuda.is_available():
                inputs = X.cuda()
                gt = Y.cuda()
            else:
                inputs, gt = X, Y

            # perform the forward pass
            outputs = unet_model(inputs)

            # compare outputs to 
            loss = criterion(outputs, gt.long())
            loss.backward()
            optimizer.step()

            # print statistics after _ minibatches
            running_loss += loss
            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))

        # compute the average loss of the epoch (accumulated over all minibatches)
        average_loss = running_loss / len(train_loader)
        losses_list.append(average_loss) # record this average loss
    
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        print(f"Average Loss: {average_loss}")


        # implement early stopping
        val_loss, _, val_ious = val(epoch) # after each epoch, perform validation
        val_losses_list.append(val_loss)
        # print('VLL', val_losses_list, len(val_losses_list))
        # print('LL', losses_list, len(losses_list))
        if val_loss < min_val_loss:
            print('Early Stopping Condition Not Reached')
            min_val_loss = val_loss
            torch.save(unet_model, 'best_model')

        # set the model back to training mode after switching to .eval() 
        # in val(epoch) function call
        unet_model.train()

    plot_losses(losses_list, val_losses_list, num_epochs, 'unet')


def plot_losses(epoch_train_loss, epoch_val_loss, num_epochs, description):

    train_loss_stds = np.zeros(num_epochs)
    train_loss_averages = np.zeros(num_epochs)

    val_loss_stds = np.zeros(num_epochs)
    val_loss_averages = np.zeros(num_epochs)

    for m in range(num_epochs):
        val_loss_averages[m] = torch.mean(epoch_val_loss[m])
        train_loss_averages[m] = torch.mean(epoch_train_loss[m])

    x = np.arange(num_epochs) + 1
    plt.clf()
    plt.plot(x, val_loss_averages, label='Validation dataset')
    plt.plot(x, train_loss_averages, label='Train dataset')
    
    np.save('plots/val_loss_averages.npy', val_loss_averages)
    np.save('plots/train_loss_averages.npy', train_loss_averages)
    
    print('val_loss_averages:', val_loss_averages)
    print('train_loss_averages:', train_loss_averages)

    plt.xlabel('Epoch')
    plt.ylabel('Normalized average cross-entropy loss')
    plt.legend(loc='best')
    filename = f"plots/{description}_loss.pdf"
    plt.savefig(filename)



def eval_model(dataloader, model):

    # eval mode can be used for validate and test
    num_samples = len(dataloader)
    # num_classes = 4

    losses = 0
    pixel_accuracies = 0
    ious = 0 
    class_ious = {0 : [0, 0], 1 : [0, 0], 2 : [0, 0], 3 : [0, 0]}

    unions = torch.zeros(1)
    intersections = torch.zeros(1)

    flattened_unions = torch.zeros(1)
    flattened_inters = torch.zeros(1)
    
    # measure how long it takes to evalueate each minibatch
    ts = time.time()
    for iter, data in enumerate(dataloader):

        X = data['scan']
        Y = data['gt']
        YM = data['gt_masks']

        if use_gpu:
            inputs = X.cuda()
            gt = Y.cuda()
            YM = YM.cuda()
        else:
            inputs, gt = X, Y

        outputs = model(inputs) # outputs = N x num_class x H x W
        loss = criterion(outputs, gt.long())

        if iter % 100 == 0:
            print("eval_model iter{}, loss: {}".format(iter, loss.item()))
        
        # accumulate errors for each minibatch
        losses += loss

        # flattened_outputs is # N x 1 x H x W
        flattened_outputs = torch.argmax(outputs, dim=1) # along class dim
        # print('flattened_outputs shape', flattened_outputs.shape)

        if iter == 0: # for first minibatch
            generate_segmented_img(flattened_outputs, 'predicted')
            generate_segmented_img(gt, 'ground_truth')

        # for each minibatch, calculate the total number of accurate pixel predictions
        pixel_accuracies += pixel_acc(flattened_outputs, gt)
        
        # also calculate the intersection over union for the minibatch
        epoch_inters, epoch_unions = iou(outputs, YM)
        
#         print('EI:', epoch_inters)
#         print('EU:', epoch_unions)

        # accumulate ious over all num_classes
        for key in class_ious.keys():
            class_ious[key][0] += epoch_inters[key]
            class_ious[key][1] += epoch_unions[key]
        
        intersections[0] += torch.nansum(epoch_inters)
        unions[0] += torch.nansum(epoch_unions)

    print("Finish eval_model, time elapsed {}\n".format(time.time() - ts))
    avg_loss = losses / num_samples
    avg_pix_acc = pixel_accuracies / num_samples
    avg_ious = intersections[0] / unions[0]

    # Compute ious
    for key in class_ious.keys():
        inter = class_ious[key][0]
        union = class_ious[key][1]
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

    seg_img = [[label_mappings[int(pix)][2][0], \
                label_mappings[int(pix)][2][1], \
                label_mappings[int(pix)][2][2]] \
                for pix in img]

    
    seg_img = np.asarray(seg_img).reshape(height, width, 3).astype(np.float32)
    plt.imsave(f"{filename}.pdf", seg_img)
    # plt.show()  


def val(epoch):

    # take output of forward pass and run through softmax
    val_model = nn.Sequential( unet_model, nn.Softmax(dim=1) )

    # put val_model in .eval() mode (https://discuss.pytorch.org/t/model-eval-vs-with-torch-nograd/19615)
    val_model.eval()

    #Calculate loss, accuracy and IoU for every epoch
    # Make sure to include a softmax after the output from your model
    print('\n-------- Evaluating validation set --------')

    # https://pytorch.org/docs/stable/generated/torch.no_grad.html
    with torch.no_grad():
        avg_val_loss, avg_pix_acc, avg_ious = eval_model(val_loader, val_model)
    
    print('\nval at epoch{}: {} loss, {} pix acc, {} iou'.format(epoch, avg_val_loss, avg_pix_acc, avg_ious))
    return avg_val_loss, avg_pix_acc, avg_ious
    
    
def test():
    test_model = torch.load("best_model")
    test_model = nn.Sequential( test_model, nn.Softmax(dim=1) )
    test_model.eval()
    #Complete this function - Calculate accuracy and IoU 
    # Make sure to include a softmax after the output from your model
    print('\n-------- Evaluating test set --------')
    with torch.no_grad():
        _, avg_pix_acc, avg_ious = eval_model(val_loader, test_model)
#         _, avg_pix_acc, avg_ious = eval_model(test_loader, test_model)

    print('test: {} pix acc, {} iou'.format(avg_pix_acc, avg_ious))
    return avg_pix_acc, avg_ious




def evalPatient(pid, Chambers, SysDia):
    
    # load the best model
    test_model = torch.load("best_model")
    test_model = nn.Sequential( test_model, nn.Softmax(dim=1) )
    test_model.eval()
    
    # get data
    temp_name = 'patient' + pid.zfill(4)
    scan_path = dataset_path + 'training/' + temp_name + '/' + temp_name + '_' + str(Chambers) + 'CH_' + SysDia + '.mhd'
    
    scan = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(scan_path))[0]
    sample = {'scan': scan, 'gt':np.zeros_like(scan)}

    # make sure this matches data_loader
    transform = transforms.Compose([ZPCC((800, 600)), DSCALE(2), ToTensor()])
    
    sample = transform(sample)
    scan = sample['scan'].unsqueeze(0)
    
    if use_gpu:
        scan = scan.cuda()
    else:
        scan = scan
    
    flattened_output = torch.argmax(test_model(scan), dim=1) # along class dim
#     print('unique_values:', torch.unique(flattened_output))
#     generate_segmented_img(flattened_output, 'predicted_test'+str(Chambers)+SysDia)

    return flattened_output


def calculateEF(outputED, outputES, LV_label=1):
    
    oED = outputED.cpu()
    oED = np.asarray(oED[0]).flatten()
    
    oES = outputES.cpu()
    oES = np.asarray(oES[0]).flatten()
    
    countED = 0
    countES = 0
    
    # find volume of left ventricle by counting number of pixels with LV label (== 1)
    for i,pixED in enumerate(oED):
        
        if pixED == LV_label:
            countED += 1
        if oES[i] == LV_label:
            countES += 1
            
    if countED == 0:
        EF = 0
    else:
        EF = (countED - countES)/countED

    return EF
        
  
    
if __name__ == '__main__':

    ##### train the network
    train()
#     test()
    
  
    ##### estimate EF from validation set
    prediction_errors = []
    Chambers = 4
    for p in val_patients:

        # check if the patient has scans for ED and ES in validation set, if so, calulate EF and record estimate from .cfg file
        # maybe average the 2ch and 4ch view estimates of EF? Maybe take the closer one?
        
        pathES = '/home/nardolino/CAMUS/training/patient'+p+'/patient'+p+'_'+str(Chambers)+'CH_ES.mhd'
        pathED = '/home/nardolino/CAMUS/training/patient'+p+'/patient'+p+'_'+str(Chambers)+'CH_ED.mhd'
        pathCfg = '/home/nardolino/CAMUS/training/patient'+p+ '/Info_' + str(Chambers) + 'CH.cfg'
#         print(pathES,'\n',pathED,'\n',pathCfg,'\n')
        if not os.path.exists(pathES) or not os.path.exists(pathED) or not os.path.exists(pathCfg): 
            print(f'patient {p} data not avaialable')
            continue
        
        outputED = evalPatient(pid=p.lstrip('0'), Chambers=4, SysDia='ED')
        outputSD = evalPatient(pid=p.lstrip('0'), Chambers=4, SysDia='ES')

        predictedEF = calculateEF(outputED, outputSD, LV_label=1)*100

        with open(pathCfg) as cfg_file:
            data = cfg_file.read().split()
            keys = [k[:-1] for k in data[0::2]]
            values = data[1::2]
            temp_dict = dict(zip(keys,values))
            
        trueEF = float(temp_dict['LVef'])
        
        print('( patient:', p, ')', 'predictedEF:', predictedEF, 'trueEF:', trueEF)
        prediction_errors.append([trueEF-predictedEF])
        
    np.save('plots/prediction_errors.npy', prediction_errors)
    
    

    ##### generate a segmentation map for patient _ (for comparison to other learned models)
    pid_display = '0003'
    Chambers_display = 2
    SysDia_display = 'ED'
    # run patient _ through the test and save the output
    display_patient_FO = evalPatient(pid_display, Chambers_display, SysDia_display) # flattened output
    generate_segmented_img(display_patient_FO, 'plots/display_patient_'+str(Chambers_display)+SysDia_display)                
    # get the corresponding ground truth
    temp_name = 'patient' + pid_display
    gt_path = dataset_path + 'training/' + temp_name + '/' + temp_name + '_' + str(Chambers_display) + 'CH_' + SysDia_display + '_gt.mhd'
    gt = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(gt_path))[0]
    sample = {'scan': np.zeros_like(gt), 'gt': gt}
    # make sure this matches data_loader
    transform = transforms.Compose([ZPCC((800, 600)), DSCALE(2), ToTensor()])
    sample = transform(sample)
    gt_display = sample['gt'].unsqueeze(0)
    generate_segmented_img(gt_display, 'plots/gt_display_patient_'+str(Chambers_display)+SysDia_display)

    
    test()

    
    
    