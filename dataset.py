import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np

class DatasetInstantaneousFrequency(Dataset):
    def __init__(self,
                 annotations_dataframe,
                 if_dir,
                 transform=None,
                 mode = 'train',
                 frame_length = 50,
                 overlap_ratio = 0.5,
                 device='cpu'):

        self.annotations = annotations_dataframe
        self.if_dir = if_dir
        self.device = device
        self.mode = mode
        self.transform = transform
        self.frame_length = frame_length
        self.overlap_ratio = overlap_ratio

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        if self.mode == 'train':
            return self._train_getitem(index)
        elif self.mode == 'test':
            return self._test_getitem(index)
        else:
            raise ValueError('Invalid mode')
    
    def _train_getitem(self, index):
        sfcc_path = self._get_train_sfcc_path(index) 
        #print(sfcc_path)
        
        label = self._get_audio_sample_label(index)
        
        # Load SFCC
        sfcc = np.load(sfcc_path)

        # Convert SFCC to tensor and move to device
        sfcc_tensor = torch.tensor(sfcc.astype(np.float32), device=self.device).unsqueeze(dim=0)
        
        # Apply transform if available
        if self.transform:
            sfcc_tensor = self.transform(sfcc_tensor)
            
        #return sfcc_tensor, label
        return sfcc_tensor, label
    
    def _test_getitem(self, index):
        frames = self.annotations.loc[index, 'frames']
        label = self._get_audio_sample_label(index)
        sfccs = []

        for idx in range(frames):
            file_path = os.path.join(self.if_dir, self.annotations.loc[index, 'file_name']) + f'{int(idx * self.frame_length * self.overlap_ratio)}_instant_frequency.npy'
            sfcc = np.load(file_path)
            sfcc = torch.tensor(sfcc.astype(np.float32)).unsqueeze(dim=0)  # Convert to PyTorch tensor
            sfccs.append(sfcc)
        
        # sfccs = np.array(sfccs)
        
        # apply transform on the elements of the list
        if self.transform:
            transformed_sfccs = []
            for sfcc in sfccs:
                transformed_sfcc = self.transform(sfcc)  # Applying composed transform
                transformed_sfccs.append(transformed_sfcc)
            sfccs_tensor = torch.stack(transformed_sfccs)
        else:
            sfccs_tensor = torch.stack(sfccs)

        return sfccs_tensor.to(self.device), label
        # sfcc_path = self._get_test_sfcc_path(index)
        
    
    def _get_train_sfcc_path(self, index):
        sfcc_filename = self.annotations.loc[index, 'instantaneous_frequency_path']  # Assuming the filename is in the first column
        # return os.path.join(self.if_dir, sfcc_filename)
        return sfcc_filename
    
    # def _get_test_sfcc_path(self, index):
    #     sfcc_filename = self.annotations.loc[index, 'file_name']
    #     return os.path.join(self.if_dir, sfcc_filename)

    def _get_audio_sample_label(self, index):
        # Assuming the label is in the second column of the CSV file
        return int(self.annotations.loc[index, 'label'] == 'dysarthria')