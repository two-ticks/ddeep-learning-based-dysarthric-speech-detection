import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np

class TorgoSpeechWithInstantaneousFrequency(Dataset):
    def __init__(self,
                 annotations_file,
                 if_dir,
                 transform=None,
                 device='cpu'):

        self.annotations = pd.read_csv(annotations_file)
        self.if_dir = if_dir
        self.device = device
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        instantaneous_frequency_path = self._get_instantaneous_frequency_path(index) 
        #print(instantaneous_frequency_path)
        
        label = self._get_audio_sample_label(index)
        
        # Load Instantaneous Frequency
        instantaneous_frequency = np.load(instantaneous_frequency_path)

        # Convert Instantaneous Frequency to tensor and move to device
        instantaneous_frequency_tensor = torch.tensor(instantaneous_frequency.astype(np.float32), device=self.device).unsqueeze(dim=0)
        
        # Apply transform if available
        if self.transform:
            instantaneous_frequency_tensor = self.transform(instantaneous_frequency_tensor)
            
        #return instantaneous_frequency_tensor, label
        return instantaneous_frequency_tensor, label
    
    def _get_instantaneous_frequency_path(self, index):
        instantaneous_frequency_filename = self.annotations.iloc[index, 0]  # Assuming the filename is in the first column
        # return os.path.join(self.if_dir, instantaneous_frequency_filename)
        return instantaneous_frequency_filename

    def _get_audio_sample_label(self, index):
        # Assuming the label is in the second column of the CSV file
        return int(self.annotations.loc[index, 'is_dysarthria'] == 'dysarthria')