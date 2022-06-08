import os
from pathlib import Path
from sqlite3 import SQLITE_CREATE_TEMP_TABLE
import numpy as np
import matplotlib.pyplot as plt

import SimpleITK
from regex import W

from scipy import ndimage

# import rawpy
# import PIL as Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from collections import namedtuple
import torch.nn.functional as F


# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision/model/data_loader.py



# a label and all meta information
Label = namedtuple( 'Label' , [
    'name'        , 
    'level3Id'    , 
    'color'       , 
    ] )


label_mappings = [

    # labels are correct. 255 is dark, 1 is light. dont use 0
    Label(  'background'        ,    0  ,  (  55,  55,  55)   ),
    Label(  'left ventricle'    ,    1  ,  ( 225,  225, 225)   ), 
    Label(  'myocardium'        ,    2  ,  (175, 175, 175)   ),
    Label(  'left atrium'       ,    3  ,  (1, 1, 1)   ),
    
]



class SSEchoDataset(Dataset):
    def __init__(self, dataset_path, TestTrain, ImQ=['Poor','Medium','Good'], Chambers=['2','4'], SysDia=['ES','ED'], transform=None):

        print('calling SSECHODataset class ...')

        self.num_class = 4
        self.dataset_path = dataset_path + TestTrain + '/'
        self.transform = transforms.Compose([ZPCC((800, 600)), DSCALE(2), ToTensor()])
       
        patient_paths = sorted(Path(self.dataset_path).glob('patient*'))
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
                        out_paths.append([scan_path])
                    else:
                        raise Exception('TrainTest string must be "testing" or "training"')
   
        return out_paths

        
    # this occurs each time a data sample is taken
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
            
        scan_path = self.data_paths[idx][0]
        gt_path = self.data_paths[idx][1]


        scan = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(scan_path))[0]
        gt = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(gt_path))[0]

        sample = {'scan': scan, 'gt': gt}

        # perform predefined transformations on scan and gt
        if self.transform:
            sample = self.transform(sample)

        # convert ground truth into one-hot encoded vectors
        gt = sample['gt']
        h, w = gt.shape
        gt_masks = torch.zeros(self.num_class, h, w)
        for c in range(self.num_class):
            gt_masks[c][gt == c] = 1

        sample['gt_masks'] = gt_masks
         

        return sample


# for ensuring all training images are the same dimension
class ZPCC(object):
    
    """ 
    Zero pad then center crop (ZPCC) scans so that they are all of the same dimension
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        
    def __call__(self, sample):

        if isinstance(self.output_size, int):
            self.output_size = (self.output_size, self.output_size)

        scan, gt = sample['scan'], sample['gt']

        h, w = scan.shape[:2]
        pad_scan = np.pad(scan, max(max(self.output_size[0]-h,self.output_size[1]-w)+1, 0)//2)
        pad_gt = np.pad(gt, max(max(self.output_size[0]-h,self.output_size[1]-w)+1, 0)//2)

        h, w = pad_scan.shape[:2]
        starth = h // 2 - (self.output_size[0] // 2)
        startw = w // 2 - (self.output_size[1] // 2)

        CC_scan = pad_scan[starth:starth + self.output_size[0], startw:startw + self.output_size[1]]
        CC_gt = pad_gt[starth:starth + self.output_size[0], startw:startw + self.output_size[1]]

        return {'scan': CC_scan, 'gt': CC_gt}
    
    
class DSCALE(object):
    
    """
    downsample the images to save on memory
    """
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor
        
    def __call__(self, sample):        
        scan, gt = sample['scan'], sample['gt']
        return {'scan': scan[::self.scale_factor, ::self.scale_factor], 'gt': gt[::self.scale_factor, ::self.scale_factor]}
        
    
    
# for converting images into tensors  
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        scan, gt = sample['scan'], sample['gt']

        return {'scan': torch.from_numpy(scan.astype(np.float32)).unsqueeze(0),
                  'gt': torch.from_numpy(gt.astype(np.float32))}