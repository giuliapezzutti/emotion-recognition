import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# -- Different preprocessing for signals

# Data folder location

data_dir = 'data'

# Create a dataframe containing all the indexes of the samples and the labels from the reference file
# The columns in the file are the following:
# names=['Participant_id','Trial','Experiment_id','Start_time','Valence','Arousal','Dominance','Liking','Familiarity']

ref_sig_dir = str(data_dir + '/participant_ratings.csv')
ref_sig = pd.read_csv(ref_sig_dir, index_col=None, header=0)

# Define the labels for Valence and Arousal

ref_sig["Valence_label"] = ref_sig['Valence'].apply(lambda l: 1 if l > 5 else 0)
ref_sig["Arousal_label"] = ref_sig['Arousal'].apply(lambda l: 1 if l > 5 else 0)

# Create a new dataframe repeating each value 32 times: in this way, it creates the indexes for the 32 channels of the
# signal extracted from each file. Each channel is then labeled with an index from 1 to 32

newref_sig = pd.DataFrame(np.repeat(ref_sig.values, 32, axis=0))
newref_sig.columns = ref_sig.columns
ref_sig = newref_sig
channel = np.arange(1, 33, 1)
channels = np.concatenate((channel, channel))
for i in range(1278):
    channels = np.concatenate((channels, channel))
ref_sig['Channel'] = channels

# Sort in order of participant_id and experiment_id (Video number) for a better visualization and print the results in a
# file in the current folder

ref_sig = ref_sig.sort_values(by=['Participant_id', 'Experiment_id'])[
    ['Participant_id', 'Experiment_id', 'Channel', 'Valence_label', 'Arousal_label']]
# print(ref_sig)
ref_sig.to_csv("referenceoutput_es.csv")

# Divide in training (70%), validation (20%) and test (10%)

train_ref_sig, val_ref_sig = train_test_split(ref_sig, test_size=0.3, stratify=ref_sig[['Valence_label','Arousal_label']])
val_ref_sig, test_ref_sig = train_test_split(val_ref_sig, test_size=0.3, stratify=val_ref_sig[['Valence_label','Arousal_label']])

# Count the elements in the sets

print('Number of total samples: {}'.format(len(ref_sig)))
print('Number of elements in training set: {}'.format(len(train_ref_sig)))
print('Number of elements in validation set: {}'.format(len(val_ref_sig)))
print('Number of elements in test set: {}'.format(len(test_ref_sig)))

# Hyperparameters for the subsequent training

batch_size = 64
num_epochs = 20
train_steps = int(np.ceil(len(train_ref_sig)/batch_size))
val_steps = int(np.ceil(len(val_ref_sig)/batch_size))
test_steps = int(np.ceil(len(test_ref_sig)/batch_size))