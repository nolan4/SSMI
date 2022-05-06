import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import SimpleITK
from regex import W

# import rawpy
# import PIL as Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision/model/data_loader.py


class SSEchoDataset(Dataset):
    def __init__(self, dataset_path, TestTrain, ImQ=['Poor','Medium','Good'], Chambers=['2','4'], SysDia=['ES','ED'], transform=None):

        print('calling SSECHODataset class ...')

        self.dataset_path = dataset_path + TestTrain + '/'
        self.transform = transform
        
        patient_paths = Path(self.dataset_path).glob('patient*')
        self.data_paths = self.process_paths(patient_paths, TestTrain, ImQ, Chambers, SysDia)


    def __len__(self):
        return len(self.data_paths)
    
    
    def process_paths(self, patient_paths, TestTrain, ImQ, Chambers, SysDia):
        
        out_paths = []
        # iterate through patient data
        for i,f in enumerate(patient_paths):
            patient_id = str(f)[-11:]
            # print(patient_id)
            
            # include scans with 'c' chamber view 
            for c in Chambers:

                # path to .cfg file
                temp_f = str(f)  + '/Info_' + c + 'CH.cfg'
                
                # skip if no 'c' chamber scans
                if not os.path.exists(temp_f): continue
                
                # create dict from cfg file for 'c' chamber view
                with open(temp_f) as cfg_file:
                    data = cfg_file.read().split()
                    keys = [k[:-1] for k in data[0::2]]
                    values = data[1::2]
                    temp_dict = dict(zip(keys,values))

                # check Info_(c)CH.cfg to see if scan satisfies quality condition
                if temp_dict['ImageQuality'] not in ImQ: continue

                # include systole and/or diastole scans
                for sd in SysDia:

                    scan_path = str(f) + '/' + patient_id + '_' + c + 'CH_' + sd + '.mhd'
                    
                    if TestTrain == 'training':
                        gt_path = str(f) + '/' + patient_id + '_' + c + 'CH_' + sd + '_gt.mhd'
                        out_paths.append([scan_path, gt_path])
                    elif TestTrain == 'testing':
                        out_paths.append(scan_path)
                    else:
                        raise Exception('TrainTest string must be "testing" or "training"')
   
        return out_paths

        
    # this is for training data ONLY
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        scan_path = self.data_paths[idx][0]
        gt_path = self.data_paths[idx][1]
        
        scan = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(scan_path))[0]
        gt = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(gt_path))[0]
                
        sample = {'scan': scan, 'gt': gt}
        
        if self.transform:
            sample = self.transform(sample)
    
        return sample


# for ensuring all training images are the same dimension
class ZeroPad(object):
    
    """ 
    Zero pad scans so that they are all of the same dimension
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        
    def __call__(self, sample):
        scan, gt = sample['scan'], sample['gt']
        
        h, w = scan.shape[:2]
        
        # if given an int for output dims
        if isinstance(self.output_size, int):
            
            if self.output_size < h or self.output_size < w:
                print(self.output_size, (h,w))
                raise Exception('one or both of scan/gt dimensions are larger than self.output_size')
            
            pad_right = int(self.output_size - w)
            pad_bottom = int(self.output_size - h)
                
        else:
            
            if self.output_size[0] < h or self.output_size[1] < w:
                print(self.output_size, (h,w))
                raise Exception('one or both of scan/gt dimensions are larger than self.output_size')
            
            pad_right = int(self.output_size[1] - w)
            pad_bottom = int(self.output_size[0] - h)
            
        
        pad_scan = np.vstack([np.hstack([scan, np.zeros((h, pad_right))]),  np.zeros((pad_bottom, w+pad_right))])
#         print('scan_pad', np.shape(pad_scan))
        pad_gt = np.vstack([np.hstack([gt, np.zeros((h, pad_right))]),  np.zeros((pad_bottom, w+pad_right))])
#         print('gt_pad', np.shape(pad_gt))

        return {'scan': pad_scan, 'gt': pad_gt}
    
    
    
# for converting images into tensors  
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        scan, gt = sample['scan'], sample['gt']

        return {'scan': torch.from_numpy(scan.astype(np.float32)),
                'gt': torch.from_numpy(gt.astype(np.float32))}