import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# This preprocessing is common to all the other code file, except for the 'signal' processing

# Data folder location

data_dir = 'data'

# Create a reference dataframe containing all the indexes of the samples and the labels from the reference file
# The columns in the file are the following:
# names=['Participant_id','Trial','Experiment_id','Start_time','Valence','Arousal','Dominance','Liking','Familiarity']

ref_wind_dir = str(data_dir + '/participant_ratings.csv')
ref_wind = pd.read_csv(ref_wind_dir, index_col=None, header=0)

# Define the labels for Valence and Arousal

ref_wind["Valence_label"] = ref_wind['Valence'].apply(lambda l: 1 if l > 5 else 0)
ref_wind["Arousal_label"] = ref_wind['Arousal'].apply(lambda l: 1 if l > 5 else 0)

# Create a new dataframe repeating each value 14 times: in this way, it creates the indexes for the 14 time-windows of 8
# seconds that are extracted from each signal. Each window is then labeled with an index from 1 to 14

newref_wind = pd.DataFrame(np.repeat(ref_wind.values, 14, axis=0))
newref_wind.columns = ref_wind.columns
ref_wind = newref_wind
window = np.arange(1, 15, 1)
windows = np.concatenate((window, window))
for i in range(1278):
    windows = np.concatenate((windows, window))
ref_wind['Window_number'] = windows

# Sort in order of participant_id and experiment_id (Video number) for a better visualization and print the results in a
# file in the current folder

ref_wind = ref_wind.sort_values(by=['Participant_id', 'Experiment_id'])[
    ['Participant_id', 'Experiment_id', 'Window_number', 'Valence_label', 'Arousal_label']]
# print(ref_wind)
ref_wind.to_csv("referenceoutput.csv")

# Divide in training (70%), validation (20%) and test (10%)

train_ref_wind, val_ref_wind = train_test_split(ref_wind, test_size=0.3, stratify=ref_wind[['Valence_label','Arousal_label']], random_state=123)
val_ref_wind, test_ref_wind = train_test_split(val_ref_wind, test_size=0.3, stratify=val_ref_wind[['Valence_label','Arousal_label']], random_state=123)

# Common hyperparameters for the subsequent training

batch_size = 64
num_epochs = 20
train_steps = int(np.ceil(len(train_ref_wind)/batch_size))
val_steps = int(np.ceil(len(val_ref_wind)/batch_size))
test_steps = int(np.ceil(len(test_ref_wind)/batch_size))
