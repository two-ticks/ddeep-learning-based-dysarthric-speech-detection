from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torchaudio
import librosa
from torch import nn
from sklearn.model_selection import KFold
from sklearn.utils import resample

from torchsummary import summary
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
import torchvision.transforms as transforms

from dataset import UASpeechWithInstantaneousFrequency
from preprocess.preprocess_utils import calculate_instantaneous_frequency, calculate_and_save_instantaneous_frequency, is_audio_valid
from network import CNNNetworkIF
# from k_fold import k_fold_validate


# File paths
# ANNOTATIONS_FILE = "/kaggle/input/dysarthria-detection/torgo_data/data.csv"
# AUDIO_DIR = "/kaggle/input/dysarthria-detection/"

non_dysarthria_folder = '/scratch/Torgo/non_dysarthria'
dysarthria_folder = '/scratch/Torgo/dysarthria'

CLEANED_ANNOTATIONS_FILE = f'./if.csv'
MODEL_PATH = f'./cnn_if.pth'
# Define the directory to save the clipped instantaneous_frequencys
save_dir = "/scratch/Torgo/instant_frequency/"


SAMPLE_RATE = 16000
# NUM_SAMPLES = 60000

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

# test parameters
test_split = .20
SHUFFLE_DATASET = True
RANDOM_SEED = 42

# Initialize constants and dataset
n_fft = 160
win_length = 160
hop_length = win_length

# Define parameters for clipping
threshold_ms = 500 # Minimum duration of audio in milliseconds
frame_length = 50
overlap_ratio = 0.5  # You can adjust this value according to your needs

base_dir = '/scratch/TorgoIF/'

# folders = ['dysarthria_female', 'dysarthria_male', 'non_dysarthria_female', 'non_dysarthria_male']

data = {'instantaneous_frequency_path': [], 'label': []}


def create_data_loader(dataset, batch_size, shuffle):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    #for input, target in data_loader:
    for input, target in tqdm(data_loader, desc='Training', leave=False):
        
        input, target = input.to(device), target.to(device)

        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("---------------------------")
    print("Finished training")

def test(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0  # Initialize total
    with torch.no_grad():
        for input, target in tqdm(data_loader, desc='Testing', leave=False):
            input, target = input.to(device), target.to(device)
            predictions = model(input)
            _, predicted = torch.max(predictions, 1)
            #print(predicted == target)
            total += target.size(0)  # Increment total by batch size
            correct += (predicted == target).sum().item()
    accuracy = correct / total
    #print("Total: ", total)
    print(f"Accuracy on test set: {accuracy * 100:.2f}%")
    return accuracy, correct   # Return the accuracy and number of correct predictions
        
def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()

def k_fold_validate(dataset, network, device, batch_size = 128, epochs = 10, learning_rate = 0.001, k_folds = 10, seed = torch.manual_seed(42)):
  # For fold results
  results = {}

  # Define the K-fold Cross Validator
  kfold = KFold(n_splits=k_folds, shuffle=True)
  for fold, (train_indices, test_indices) in enumerate(kfold.split(dataset)):
      print(f'Fold: {fold}')
      
      train_sampler = SubsetRandomSampler(train_indices)
      test_sampler = SubsetRandomSampler(test_indices)

      train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size , 
                                                sampler=train_sampler)
      test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size ,
                                                      sampler=test_sampler)
      
      
      # initialise CNN network
      net = network().to(device)
      net.apply(reset_weights)
      
      # initialise loss funtion + optimiser
      loss_fn = nn.CrossEntropyLoss()
      # loss_fn = nn.BCEWithLogitsLoss()
      optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate)

      # train model
      train(net, train_dataloader, loss_fn, optimiser, device, epochs)
      
      # save the model
      save_path = f'./model-fold-{fold}.pth'
      torch.save(net.state_dict(), save_path)
      
      # test model
      accuracy, _ = test(net, test_dataloader, device)
      results[fold] = accuracy
      

  sum = 0.0
  for key, value in results.items():
      print(f'Fold {key}: {value} %')
      sum += value
  print(f'Average: {100.0 * sum/len(results.items())} %')

  return 100.0 * sum/len(results.items()), results


class FLAG:
    CLEAN = True
    CLIP = True
    BALANCE = True
    TRANSFORM = True
    TRAIN = True



if __name__ == "__main__":
  
#   # prepare data
#   data = []
#   for folder_path, label in [(non_dysarthria_folder, 'non-dysarthria'), (dysarthria_folder, 'dysarthria')]:
#     for root, directories, files in os.walk(folder_path):
#         for file in files:
#             if file.lower().endswith('.wav'):
#                 file_path = os.path.join(root, file)
#                 data.append({'file_path': file_path, 'label': label})

#   raw_data = pd.DataFrame(data)
#   raw_data.to_csv(CLEANED_ANNOTATIONS_FILE, index=False)

#   print(raw_data)


#   # clean data
#   rows_to_delete = []
#   for index, row in tqdm(raw_data.iterrows(), total=len(raw_data), desc="cleaning data"):
#     audio_path = row['file_path']
#     if not is_audio_valid(audio_path, threshold_ms):
#         rows_to_delete.append(index)

#   print("Cleaned data")
#   # Delete marked rows
#   cleaned_data = raw_data.drop(index=rows_to_delete)

#   # Write the cleaned data back to the CSV file
#   cleaned_data.to_csv(CLEANED_ANNOTATIONS_FILE, index=False)

#   # print("Cleaned data")
#   print(cleaned_data)


  INSTANT_FREQUENCY_DIR = "/scratch/Torgo/instant_frequency"
#   os.makedirs(INSTANT_FREQUENCY_DIR, exist_ok=True)

#   # Add instantaneous frequency information to cleaned annotations
#   instantaneous_frequency_paths = []
#   labels = []

#   feature_dataframe = pd.DataFrame()
  
#   for index, row in tqdm(cleaned_data.iterrows(), total=len(cleaned_data), desc="clip and save IF"):
#     # print("in loop")
#     # print(row)
#     audio_path = row['file_path']
#     label = row['label']
#     # Calculate and save instantaneous frequency
#     instant_frequency_paths = calculate_and_save_instantaneous_frequency(audio_path, n_fft, hop_length, overlap_ratio, frame_length, INSTANT_FREQUENCY_DIR)
#     # magnitude_representation_paths = calculate_and_save_magnitude(audio_path, n_fft, hop_length, MAGNITUDE_DIR)
        
#     instantaneous_frequency_paths.extend(instant_frequency_paths)
#     # magnitude_paths.extend(magnitude_representation_paths)
#     labels.extend([label] * len(instant_frequency_paths))
      
#   # Add the instantaneous frequency paths to the cleaned data DataFrame
#   feature_dataframe['instantaneous_frequency_path'] = instantaneous_frequency_paths
#   feature_dataframe['is_dysarthria'] = labels
#   feature_dataframe.to_csv(CLEANED_ANNOTATIONS_FILE, index=False)

#   print(feature_dataframe)

#   data = []

#   # clip and save instantaneous frequency
#   for index, row in feature_dataframe.iterrows():
#     instantaneous_frequency_path = row['instantaneous_frequency_path']
#     label = row['is_dysarthria']
#     clipped_instantaneous_frequency_data = clip_and_save_if(instantaneous_frequency_path, label, save_dir, frame_length, overlap_ratio)
#     data.extend(clipped_instantaneous_frequency_data)

  
  
  
  
#   for folder in folders:
#       if_dir = os.path.join(base_dir, folder)
#       if_files = [f for f in os.listdir(if_dir) if os.path.isfile(os.path.join(if_dir, f))]
      
#       for if_file in tqdm(if_files, desc="cleaning if"):
#           instantaneous_frequency = np.loadtxt(os.path.join(if_dir, if_file))
#           if len(instantaneous_frequency.shape) == 2 and instantaneous_frequency.shape[1] > frame_length: # discard empty or short instantaneous_frequency files 
#               data['instantaneous_frequency_path'].append(os.path.join(if_dir, if_file))
#               if 'non_dysarthria' in folder:
#                   data['label'].append('non_dysarthria')
#               else:
#                   data['label'].append('dysarthria')

#   # Create a DataFrame
# #   instantaneous_frequency_dataframe = pd.DataFrame(data)

#   # Save the DataFrame to a CSV file
#   instantaneous_frequency_dataframe.to_csv('instantaneous_frequency_data.csv', index=False)
#   print(instantaneous_frequency_dataframe)

#   # Create the directory if it does not exist
#   os.makedirs(save_dir, exist_ok=True)


#   # List to accumulate data for CLEANED_ANNOTATIONS_FILE
#   cleaned_data = []

#   # Iterate through each row in the DataFrame and clip the instantaneous_frequencys
#   for index, row in tqdm(instantaneous_frequency_dataframe.iterrows(), total=len(instantaneous_frequency_dataframe), desc="preprocessing instantaneous_frequency"):
#       instantaneous_frequency_path = row['instantaneous_frequency_path']
#       label = row['label']
#       clipped_instantaneous_frequency_data = calculate_and_save_instantaneous_frequency(instantaneous_frequency_path, label, save_dir, frame_length, overlap_ratio)
#       cleaned_data.extend(clipped_instantaneous_frequency_data)

  # Create a DataFrame from cleaned_data
#   cleaned_df = pd.DataFrame(cleaned_data)
#   cleaned_df = feature_dataframe
  
  cleaned_df = pd.read_csv('if.csv')


  # Save the DataFrame to a CSV file
  cleaned_df.to_csv(CLEANED_ANNOTATIONS_FILE, index=False)


  print(cleaned_df)
  label_counts = cleaned_df['is_dysarthria'].value_counts()
  # print(label_counts)

  # Calculate the percentage of each label
  total_samples = len(cleaned_df)
  percentage_dysarthria = (label_counts.get('dysarthria', 0) / total_samples) * 100
  percentage_non_dysarthria = (label_counts.get('non-dysarthria', 0) / total_samples) * 100

  # Print the results
  print(f"Percentage of samples labeled as 'dysarthria': {percentage_dysarthria:.2f}%")
  print(f"Percentage of samples labeled as 'non_dysarthria': {percentage_non_dysarthria:.2f}%")

  # balance the dataset


  # Assuming cleaned_df is your DataFrame containing the cleaned annotations

  # Separate majority and minority classes
  majority_class = cleaned_df[cleaned_df['is_dysarthria'] == 'non-dysarthria']
  minority_class = cleaned_df[cleaned_df['is_dysarthria'] == 'dysarthria']

  # Upsample minority class to match the number of samples in the majority class
  minority_upsampled = resample(minority_class, 
                              replace=True,  # Sample with replacement
                              n_samples=len(majority_class),  # Match the number of samples in the majority class
                              random_state=42)  # Set random state for reproducibility

  # Concatenate majority class with upsampled minority class
  balanced_df = pd.concat([majority_class, minority_upsampled])

  

  # Save the balanced DataFrame to a CSV file
  balanced_df.to_csv(CLEANED_ANNOTATIONS_FILE, index=False)

  # Print the balanced DataFrame
  print(balanced_df)

  # Print label counts after balancing
  label_counts = balanced_df['is_dysarthria'].value_counts()
  print(label_counts)

  # Calculate the percentage of each label
  total_samples = len(balanced_df)
  percentage_dysarthria = (label_counts.get('dysarthria', 0) / total_samples) * 100
  percentage_non_dysarthria = (label_counts.get('non-dysarthria', 0) / total_samples) * 100

  print("Percentage of dysarthria samples:", percentage_dysarthria)
  print("Percentage of non-dysarthria samples:", percentage_non_dysarthria)


  PROCESSED_if_dir = save_dir


  if torch.cuda.is_available():
      device = "cuda"
  else:
      device = "cpu"
  print(f"Using device {device}")

  torgo_with_instantaneous_frequency = UASpeechWithInstantaneousFrequency(annotations_file=CLEANED_ANNOTATIONS_FILE, if_dir=PROCESSED_if_dir, device="cpu") # NOTE: load data on CPU then shift to GPU


  print(torgo_with_instantaneous_frequency[len(torgo_with_instantaneous_frequency)-1])
  print(torgo_with_instantaneous_frequency[1])


  # Creating data indices for training and test splits:
  dataset_size = len(torgo_with_instantaneous_frequency)
  indices = list(range(dataset_size))
  split = int(np.floor(test_split * dataset_size))

  if SHUFFLE_DATASET :
      np.random.seed(RANDOM_SEED)
      np.random.shuffle(indices)
  train_indices, test_indices = indices[split:], indices[:split]

  # Creating PT data samplers and loaders:
  train_sampler = SubsetRandomSampler(train_indices)
  test_sampler = SubsetRandomSampler(test_indices)
  
  print("samplers")
  # NOTE: load data on CPU then shift to GPU
  loader = torch.utils.data.DataLoader(torgo_with_instantaneous_frequency, batch_size=1024, num_workers = 4)
  
  sums = 0
  squares = 0
  total_samples = 0

  # Iterate over the DataLoader to compute sums and counts
  for batch in tqdm(loader, desc="computing mean and std", leave=False):
      data = batch[0]  # Assuming your data is in the first element of the batch tuple
    
      # Update sums and counts
      sums += data.sum(dim=0)
      squares += (data ** 2).sum(dim=0)
      total_samples += data.size(0)

  # Compute mean and standard deviation
  mean = sums / total_samples
  variance = (squares / total_samples) - (mean ** 2)
  std = torch.sqrt(variance)

  # Add a small epsilon to avoid division by zero in standard deviation
  std += 1.0e-8

  #data = next(iter(loader))
  print("finding mean")

  #mean = data[0].mean(axis=0) 
  #std = data[0].std(axis=0) + 1.0e-8

  print(mean)
  print(std)

  composed = transforms.transforms.Normalize(mean, std, inplace=False)

  torgo_with_instantaneous_frequency_transformed = UASpeechWithInstantaneousFrequency(annotations_file=CLEANED_ANNOTATIONS_FILE, if_dir=PROCESSED_if_dir, transform = composed, device = "cpu")

  print(data[0].shape)


  train_dataloader = torch.utils.data.DataLoader(torgo_with_instantaneous_frequency_transformed, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers = 4)
  test_dataloader  = torch.utils.data.DataLoader(torgo_with_instantaneous_frequency_transformed, batch_size=BATCH_SIZE, sampler=test_sampler, num_workers = 4)

  cnn_instantaneous_frequency = CNNNetworkIF().to(device)
  summary(cnn_instantaneous_frequency.to(device), (1, 81, 50))

  # initialise loss funtion + optimiser
  loss_fn = nn.CrossEntropyLoss()
  optimiser = torch.optim.Adam(cnn_instantaneous_frequency.parameters(), lr=LEARNING_RATE)
  
  print("train model")
  # train model
  if os.path.exists(MODEL_PATH):
    print("Model already exists, skipping training")
    state_dict = torch.load(MODEL_PATH)
    cnn_instantaneous_frequency.load_state_dict(state_dict)
  else: 
    train(cnn_instantaneous_frequency, train_dataloader, loss_fn, optimiser, device, EPOCHS)


  # class_mapping = ["non-dysarthia", "dysarthia"]

  # input_signal, target = torgo_with_instantaneous_frequency_transformed[5][0].to(device), torgo_with_instantaneous_frequency_transformed[5][1] # [num channels, fr, time] but cnn expects [batchsize, num channels, fr, time]
  # input_signal.unsqueeze_(0) #inplace, notice uncsqueeze_(0) means it is applied at 0th index

  # print(cnn_instantaneous_frequency(input_signal))

  # # make an inference
  # predicted, expected = predict(cnn_instantaneous_frequency, input_signal, target,
  #                               class_mapping)
  # print(f"Predicted: '{predicted}', expected: '{expected}'")


  # test 

  test(cnn_instantaneous_frequency, test_dataloader, device)

  torch.save(cnn_instantaneous_frequency.state_dict(), MODEL_PATH)

  # turn True for K_Fold 
  if True:
    k_fold_validate(dataset=torgo_with_instantaneous_frequency_transformed, network=CNNNetworkIF, device=device)

