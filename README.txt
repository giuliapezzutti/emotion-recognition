PEZZUTTI GIULIA

The Python code present in this folder is structured as follows:

- Global: in this file, a pre-processing for the subsequent data loading is present. This file is then included into the other files, except for Signal.
- Global_signal contains the pre-processing needed for the signal dataset and will be imported only in that file.

- Functions: this file contains all the function used in the analysis. The first ones are needed for load windowed data or entire signal. Then, all the functions needed for each
data loading and subsequent feature extraction are reported. At the end, the two functions for the creations of the two types of dataset (data-label and data-data for autoencoder) 
are built. The last function allows to predict data passed in input, calculate some score and print plots of the training.

- Window, FFT, ICA, PCA, KPCA, PCC, SC files contain in lines 12 - 174 the computation of all the networks for Arousal, 175 - 314 the networks for Valence and 315 - 452 the networks
for both labels. 

- Signal contains a different initial preprocessing and then the computation of the networks: 78 - 240 for Arousal, 241 - 378 for Valence and 379 - 514 for both.

- Demo contains the code used for the demo during the presentation