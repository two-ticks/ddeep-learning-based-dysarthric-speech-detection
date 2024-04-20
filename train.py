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
import copy

from torchsummary import summary
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
import torchvision.transforms as transforms

from dataset import TorgoSpeechWithInstantaneousFrequency
from preprocess import calculate_and_save_instantaneous_frequency, is_audio_valid
from network import CNNNetworkIF

# File paths
# ANNOTATIONS_FILE = "/kaggle/input/dysarthria-detection/torgo_data/data.csv"
# AUDIO_DIR = "/kaggle/input/dysarthria-detection/"

non_dysarthria_folder = '/scratch/Torgo/non_dysarthria'
dysarthria_folder = '/scratch/Torgo/dysarthria'

CLEANED_ANNOTATIONS_FILE = f'./if.csv'
CLIPPED_ANNOTATIONS_FILE = f'./clipped_if.csv'

MODEL_PATH = f'./cnn_if.pth'
# Define the directory to save the clipped instantaneous_frequencys
INSTANT_FREQUENCY_DIR = "/scratch/Torgo/instant_frequency/"


SAMPLE_RATE = 16000
# NUM_SAMPLES = 60000

BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 0.01

# test parameters
test_split = .20
validation_split = .05
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


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.status = ""

    def __call__(self, model, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
        elif self.best_loss - val_loss >= self.min_delta:
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_loss = val_loss
            self.counter = 0
            self.status = f"Improvement found, counter reset to {self.counter}"
        else:
            self.counter += 1
            self.status = f"No improvement in the last {self.counter} epochs"
            if self.counter >= self.patience:
                self.status = f"Early stopping triggered after {self.counter} epochs."
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model)
                return True
        return False


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    model.train()
    total_loss = 0
    #for input, target in data_loader:
    for input, target in tqdm(data_loader, desc='Training', leave=False):
        # calculate loss
        prediction = model(input.to(device))
        loss = loss_fn(prediction, target.to(device))

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        total_loss += loss.item()

    # print(f"loss: {total_loss:.2f}")
    # return total_loss / len(data_loader)
    return loss.item()


def validate_single_epoch(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    for input, target in tqdm(data_loader, desc='Validation', leave=False):
        with torch.no_grad():
            prediction = model(input.to(device))
            loss = loss_fn(prediction, target.to(device))
            total_loss += loss.item()
    # return total_loss / len(data_loader)
    return loss.item()
        
        
def train_validate(model, train_data_loader, validation_data_loader, loss_fn, optimiser, device, epochs):
    done = False
    epoch = 0
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)
    LEARNING_RATE = optimiser.param_groups[0]['lr']
    while not done and epoch < epochs:
        epoch += 1
        print(f"Epoch {epoch}")
        training_loss = train_single_epoch(model, train_data_loader, loss_fn, optimiser, device)
        validation_loss = validate_single_epoch(model, validation_data_loader, loss_fn, device)
        print(f"Training loss: {training_loss:.2f}", f" Validation loss: {validation_loss:.2f}")
        if early_stopping(model, validation_loss):
            if LEARNING_RATE > 1.0e-6:
                LEARNING_RATE /= 2
                optimiser.param_groups[0]['lr'] = LEARNING_RATE
            else :
                done = True
        print("---------------------------")
    print("Finished training")

def test(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0  # Initialize total
    print(len(data_loader))
    with torch.no_grad():
        for input_batch, target_batch in tqdm(data_loader, desc='Testing', leave=False):
          # print("input_batch", len(input_batch), len(target_batch))
          correct_batch = 0
          for input, target in zip(input_batch, target_batch):  
              input, target = input.to(device), target
              # print(input.shape, target)
              # print(len(input), target)

              # predicted_batch = torch.zeros(target.size(0))
              # for input in input_batch:
              #     input = input.to(device)

              predictions = model(input)
              # print(predictions)
              _, predicted = torch.max(predictions, 1)
              # majority voting
              # predicted.append(sum(predicted) > len(predicted) / 2)
                  
              # print(predicted == target)
              # print("predicted", int(sum(predicted) > len(predicted) / 2), target)
              correct_batch += (int(sum(predicted) > len(predicted) / 2) == target)
              
          total += len(target_batch)
          correct += correct_batch
          # print(f"batch details: {correct_batch}/{len(target_batch)}")
    accuracy = correct / total
    #print("Total: ", total)
    print(f"Accuracy on test set: {accuracy * 100:.2f}%")
    print(f"Correct: {correct}")
    print(f"Total: {total}")
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

def k_fold_validate(dataset, network, device, INSTANT_FREQUENCY_DIR, frame_length = 50, overlap_ratio = 0.5, batch_size = 128, epochs = 5, learning_rate = 0.001, k_folds = 10, seed = torch.manual_seed(42)):
  # For fold results
  results = {}

  # Define the K-fold Cross Validator
  kfold = KFold(n_splits=k_folds, shuffle=True)
  for fold, (train_indices, test_indices) in enumerate(kfold.split(dataset)):
      
      test_data = dataset.iloc[test_indices].reset_index(drop=True)
      train_data = dataset.iloc[train_indices].reset_index(drop=True)
      
      
      print(f'Fold: {fold}')

      # from train_data create train_frames_data
      train_frames_list = []
      for index, row in tqdm(train_data.iterrows(), total=len(train_data), desc="train frames"):
        label = row['label']
        frames = row['frames']
        for idx in range(frames):
            file_name = os.path.join(INSTANT_FREQUENCY_DIR, row['file_name']) + f'{int(idx * frame_length * overlap_ratio)}_instant_frequency.npy'
            # file_name = row['file_name'] + f'{idx * frame_length * overlap_ratio}_instant_frequency.npy'
            train_frames_list.append({'instantaneous_frequency_path': os.path.join(INSTANT_FREQUENCY_DIR, file_name), 'label': label}) 

      train_frames_data = pd.DataFrame(train_frames_list)
      print(train_frames_data)

      torgo_with_instantaneous_frequency_train = TorgoSpeechWithInstantaneousFrequency(annotations_dataframe=train_frames_data, if_dir=INSTANT_FREQUENCY_DIR, mode= 'train', device="cpu") # NOTE: load data on CPU then shift to GPU
      torgo_with_instantaneous_frequency_test = TorgoSpeechWithInstantaneousFrequency(annotations_dataframe=test_data, if_dir=INSTANT_FREQUENCY_DIR, mode= 'test', frame_length=frame_length, overlap_ratio=overlap_ratio, device="cpu") # NOTE: load data on CPU then shift to GPU

      
      # NOTE: load data on CPU then shift to GPU
      loader = torch.utils.data.DataLoader(torgo_with_instantaneous_frequency_train, batch_size=1024, num_workers = 4)
      
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

      torgo_with_instantaneous_frequency_train_transformed = TorgoSpeechWithInstantaneousFrequency(annotations_dataframe=train_frames_data, if_dir=INSTANT_FREQUENCY_DIR, transform = composed, mode='train', device = "cpu")
      torgo_with_instantaneous_frequency_test_transformed = TorgoSpeechWithInstantaneousFrequency(annotations_dataframe=test_data, if_dir=INSTANT_FREQUENCY_DIR, transform = composed, mode='test', device = "cpu")

      # print(data[0].shape)

      




      train_dataloader = torch.utils.data.DataLoader(torgo_with_instantaneous_frequency_train_transformed, batch_size=BATCH_SIZE, shuffle=True, num_workers = 4)
      test_dataloader  = torch.utils.data.DataLoader(torgo_with_instantaneous_frequency_test_transformed, batch_size=BATCH_SIZE, shuffle=True, num_workers = 4, collate_fn=collate_fn)
      
      
      # initialise CNN network
      net = network().to(device)
      net.apply(reset_weights)
      
      # initialise loss funtion + optimiser
      loss_fn = nn.CrossEntropyLoss()
      # loss_fn = nn.BCEWithLogitsLoss()
      optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate)

      # train model
      train_validate(net, train_dataloader, validation_dataloader, loss_fn, optimiser, device, epochs)
      
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



def collate_fn(batch):
    inputs = [sample[0] for sample in batch]
    targets = [sample[1] for sample in batch]
    # print("collate_fn", len(inputs), len(targets))
    return [inputs, targets]


# FLAG as Dictionary

class FLAG:
    CLEAN = False
    CLIP = False
    DRY_RUN = False
    BALANCE = True
    TRANSFORM = True
    TRAIN = True
   

if __name__ == "__main__":
  
  if FLAG.CLEAN:
    # prepare data
    data = []
    for folder_path, label in [(non_dysarthria_folder, 'non-dysarthria'), (dysarthria_folder, 'dysarthria')]:
        for root, directories, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith('.wav'):
                    file_path = os.path.join(root, file)
                    data.append({'file_path': file_path, 'file_name': file, 'label': label})

    raw_data = pd.DataFrame(data)
    raw_data.to_csv(CLEANED_ANNOTATIONS_FILE, index=False)

    print(raw_data)


    # clean data
    rows_to_delete = []
    for index, row in tqdm(raw_data.iterrows(), total=len(raw_data), desc="cleaning data"):
        audio_path = row['file_path']
        if not is_audio_valid(audio_path, threshold_ms):
            rows_to_delete.append(index)

    print("Cleaned data")
    # Delete marked rows
    cleaned_data = raw_data.drop(index=rows_to_delete)

    # Write the cleaned data back to the CSV file
    cleaned_data.to_csv(CLEANED_ANNOTATIONS_FILE, index=False)

  cleaned_data = pd.read_csv(CLEANED_ANNOTATIONS_FILE)
  # print("Cleaned data")
  # print(cleaned_data)

  if FLAG.DRY_RUN:
    # select only 100 samples for testing from start and 100 from end
    cleaned_data = pd.concat([cleaned_data.head(500), cleaned_data.tail(500)])
    print(cleaned_data)


  #   cleaned_data = cleaned_data.head(1000)
  

  if FLAG.CLIP:
    
    os.makedirs(INSTANT_FREQUENCY_DIR, exist_ok=True)
    # Add instantaneous frequency information to cleaned annotations
    instantaneous_frequency_paths = []
    labels = []

    frames = []

    feature_dataframe = pd.DataFrame()
    
    for index, row in tqdm(cleaned_data.iterrows(), total=len(cleaned_data), desc="clip and save IF"):
        # print("in loop")
        # print(row)
        audio_path = row['file_path']
        label = row['label']
        # Calculate and save instantaneous frequency
        instant_frequency_paths = calculate_and_save_instantaneous_frequency(audio_path, n_fft, hop_length, overlap_ratio, frame_length, INSTANT_FREQUENCY_DIR)
        # magnitude_representation_paths = calculate_and_save_magnitude(audio_path, n_fft, hop_length, MAGNITUDE_DIR)
            
        instantaneous_frequency_paths.extend(instant_frequency_paths)
        # magnitude_paths.extend(magnitude_representation_paths)
        labels.extend([label] * len(instant_frequency_paths))
        frames.append(len(instant_frequency_paths))
        
    # Add the instantaneous frequency paths to the cleaned data DataFrame
    feature_dataframe['instantaneous_frequency_path'] = instantaneous_frequency_paths
    feature_dataframe['is_dysarthria'] = labels
    feature_dataframe.to_csv(CLIPPED_ANNOTATIONS_FILE, index=False)

    # print(feature_dataframe)

    # add the frames coloumn to the cleaned data
    cleaned_data['frames'] = frames

    print(cleaned_data)
    # cleaned_data.to_csv(CLEANED_ANNOTATIONS_FILE, index=False)
    cleaned_data.to_csv('cleaned_data_framecount.csv', index=False)

  cleaned_df = pd.read_csv('cleaned_data_framecount.csv')

  # Save the DataFrame to a CSV file
  # cleaned_df.to_csv(CLEANED_ANNOTATIONS_FILE, index=False)


  print(cleaned_df)
  label_counts = cleaned_df['label'].value_counts()
  # print(label_counts)

  # Calculate the percentage of each label
  total_samples = len(cleaned_df)
  percentage_dysarthria = (label_counts.get('dysarthria', 0) / total_samples) * 100
  percentage_non_dysarthria = (label_counts.get('non-dysarthria', 0) / total_samples) * 100

  # Print the results
  print(f"Percentage of samples labeled as 'dysarthria': {percentage_dysarthria:.2f}%")
  print(f"Percentage of samples labeled as 'non_dysarthria': {percentage_non_dysarthria:.2f}%")

  # BALANCE

  if FLAG.BALANCE:
    # Assuming cleaned_df is your DataFrame containing the cleaned annotations

    # Separate majority and minority classes
    majority_class = cleaned_df[cleaned_df['label'] == 'non-dysarthria']
    minority_class = cleaned_df[cleaned_df['label'] == 'dysarthria']

    # Upsample minority class to match the number of samples in the majority class
    minority_upsampled = resample(minority_class, 
                                replace=True,  # Sample with replacement
                                n_samples=len(majority_class),  # Match the number of samples in the majority class
                                random_state=42)  # Set random state for reproducibility

    # Concatenate majority class with upsampled minority class
    cleaned_df = pd.concat([majority_class, minority_upsampled])

    # Save the balanced DataFrame to a CSV file
    # balanced_df.to_csv(BALANCED_ANNOTATIONS_FILE, index=False)

    # Print the balanced DataFrame
    print(cleaned_df)

    # Print label counts after balancing
    label_counts = cleaned_df['label'].value_counts()
    print(label_counts)

    # Calculate the percentage of each label
    total_samples = len(cleaned_df)
    percentage_dysarthria = (label_counts.get('dysarthria', 0) / total_samples) * 100
    percentage_non_dysarthria = (label_counts.get('non-dysarthria', 0) / total_samples) * 100

    print("Percentage of dysarthria samples:", percentage_dysarthria)
    print("Percentage of non-dysarthria samples:", percentage_non_dysarthria)

    # cleaned_df = balanced_df


  # TRAINING, TESTING AND INFERENCE

  if torch.cuda.is_available():
      device = "cuda"
  else:
      device = "cpu"
  print(f"Using device {device}")

  # split the data into training and testing

  test_data = cleaned_df.sample(frac=test_split, random_state=RANDOM_SEED).reset_index(drop=True)
  train_data = cleaned_df.drop(test_data.index).reset_index(drop=True)

  # print(train_data)
  # print("test data")
  # print(test_data)

  # from train_data create train_frames_data
  train_frames_list = []
  for index, row in tqdm(train_data.iterrows(), total=len(train_data), desc="train frames"):
    label = row['label']
    frames = row['frames']
    for idx in range(frames):
        file_name = os.path.join(INSTANT_FREQUENCY_DIR, row['file_name']) + f'{int(idx * frame_length * overlap_ratio)}_instant_frequency.npy'
        # file_name = row['file_name'] + f'{idx * frame_length * overlap_ratio}_instant_frequency.npy'
        train_frames_list.append({'instantaneous_frequency_path': os.path.join(INSTANT_FREQUENCY_DIR, file_name), 'label': label}) 

  train_frames_data = pd.DataFrame(train_frames_list)
  print(train_frames_data)

  validation_frames_data = train_frames_data.sample(frac=validation_split, random_state=RANDOM_SEED).reset_index(drop=True)
  train_frames_data = train_frames_data.drop(validation_frames_data.index).reset_index(drop=True)

  torgo_with_instantaneous_frequency_train = TorgoSpeechWithInstantaneousFrequency(annotations_dataframe=train_frames_data, if_dir=INSTANT_FREQUENCY_DIR, mode= 'train', device="cpu") # NOTE: load data on CPU then shift to GPU
  torgo_with_instantaneous_frequency_test = TorgoSpeechWithInstantaneousFrequency(annotations_dataframe=test_data, if_dir=INSTANT_FREQUENCY_DIR, mode= 'test', frame_length=frame_length, overlap_ratio=overlap_ratio, device="cpu") # NOTE: load data on CPU then shift to GPU
  # TODO : add validation mode to the dataset
  
  torgo_with_instantaneous_frequency_validation = TorgoSpeechWithInstantaneousFrequency(annotations_dataframe=validation_frames_data, if_dir=INSTANT_FREQUENCY_DIR, mode= 'train', frame_length=frame_length, overlap_ratio=overlap_ratio, device="cpu") # NOTE: load data on CPU then shift to GPU
  
  # print("test len", len(torgo_with_instantaneous_frequency_test))
  
  

#   torgo_with_instantaneous_frequency = TorgoSpeechWithInstantaneousFrequency(annotations_file=CLEANED_ANNOTATIONS_FILE, if_dir=INSTANT_FREQUENCY_DIR, device="cpu") # NOTE: load data on CPU then shift to GPU


  print(torgo_with_instantaneous_frequency_train[len(torgo_with_instantaneous_frequency_train)-1])
  print(torgo_with_instantaneous_frequency_train[1])


#   # Creating data indices for training and test splits:
#   dataset_size = len(torgo_with_instantaneous_frequency_train)
#   indices = list(range(dataset_size))
#   split = int(np.floor(test_split * dataset_size))

#   if SHUFFLE_DATASET :
#       np.random.seed(RANDOM_SEED)
#       np.random.shuffle(indices)
#   train_indices, test_indices = indices[split:], indices[:split]

#   # Creating PT data samplers and loaders:
#   train_sampler = SubsetRandomSampler(train_indices)
#   test_sampler = SubsetRandomSampler(test_indices)
  
#   print("samplers")
  
  # NOTE: load data on CPU then shift to GPU
  loader = torch.utils.data.DataLoader(torgo_with_instantaneous_frequency_train, batch_size=1024, num_workers = 4)
  
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

  torgo_with_instantaneous_frequency_train_transformed = TorgoSpeechWithInstantaneousFrequency(annotations_dataframe=train_frames_data, if_dir=INSTANT_FREQUENCY_DIR, transform = composed, mode='train', device = "cpu")
  torgo_with_instantaneous_frequency_test_transformed = TorgoSpeechWithInstantaneousFrequency(annotations_dataframe=test_data, if_dir=INSTANT_FREQUENCY_DIR, transform = composed, mode='test', device = "cpu")
  torgo_with_instantaneous_frequency_validation_transformed = TorgoSpeechWithInstantaneousFrequency(annotations_dataframe=validation_frames_data, if_dir=INSTANT_FREQUENCY_DIR, transform = composed, mode='train', device = "cpu")

  print(data[0].shape)

  train_dataloader = torch.utils.data.DataLoader(torgo_with_instantaneous_frequency_train_transformed, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATASET, num_workers = 4)
  test_dataloader  = torch.utils.data.DataLoader(torgo_with_instantaneous_frequency_test_transformed, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATASET, num_workers = 4, collate_fn=collate_fn)
  validation_dataloader  = torch.utils.data.DataLoader(torgo_with_instantaneous_frequency_validation_transformed, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATASET, num_workers = 4)

  print("test dataloader", len(test_dataloader))

  cnn_instantaneous_frequency = CNNNetworkIF().to(device)

  summary(cnn_instantaneous_frequency.to(device), (1, 81, 50))

  # initialise loss funtion + optimiser
  loss_fn = nn.CrossEntropyLoss()
#   optimiser = torch.optim.Adam(cnn_instantaneous_frequency.parameters(), lr=LEARNING_RATE)
  optimiser = torch.optim.SGD(cnn_instantaneous_frequency.parameters(), lr=LEARNING_RATE, momentum=0.9)
  
  # TODO: use compile to exploit JIT
  
  print("train model")
  # train model
  if False: #os.path.exists(MODEL_PATH):
    print("Model already exists, skipping training")
    state_dict = torch.load(MODEL_PATH)
    cnn_instantaneous_frequency.load_state_dict(state_dict)
  else: 
    train_validate(cnn_instantaneous_frequency, train_data_loader = train_dataloader, validation_data_loader=validation_dataloader, loss_fn=loss_fn, optimiser=optimiser, device=device, epochs=EPOCHS)


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

  # torch.save(cnn_instantaneous_frequency.state_dict(), MODEL_PATH)

  # turn True for K_Fold 
  if True:
    k_fold_validate(dataset=cleaned_df, network=CNNNetworkIF, device=device, INSTANT_FREQUENCY_DIR=INSTANT_FREQUENCY_DIR, epochs=EPOCHS)

