from Global import *
from Global_signal import *
import tensorflow as tf
import numpy as np
import os
from scipy.signal import medfilt
import scipy
from sklearn.preprocessing import scale
from sklearn.decomposition import *
import _pickle as cPickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.fft import rfft
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc, accuracy_score

mpl.rcParams['figure.figsize'] = (10, 10)
mpl.rcParams['axes.grid'] = True
mpl.rcParams['legend.fontsize'] = 'large'


def load_data(index, data_dir):

    # Given an index of the reference table, extract participant (# file), experiment number and time (number of window)

    participant_id = int(ref_wind.iloc[int(index)]['Participant_id'])
    experiment_id = int(ref_wind.iloc[int(index)]['Experiment_id'] - 1)
    window_number = int(ref_wind.iloc[int(index)]['Window_number'] - 1)

    # Decode needed for the subsequent data loading and extraction of the correspondent file name

    if isinstance(data_dir, bytes):
        data_dir = data_dir.decode()
    if isinstance(participant_id, bytes):
        participant_id = participant_id.decode()
    if int(participant_id) < 10:
        file_name = str('s0' + str(participant_id))
    else:
        file_name = str('s' + str(participant_id))

    # File path creation - data loading, considering only 'data' part, all the 32 channels and from 3rd second
    # (with a sampling rate of 128, corresponds from value 384) - extraction of the correspondent window

    file_dat = str(data_dir + '/' + str(file_name) + '.dat')
    data = cPickle.load(open(file_dat, 'rb'), encoding='latin1')['data'][experiment_id, 0:32, 384:]
    data = data[:, window_number * 128:((window_number + 8) * 128)]

    return data


def load_signal(index, data_dir):

    # Given an index of the reference table, extract participant (# file), experiment number and channel

    participant_id = int(ref_sig.iloc[int(index)]['Participant_id'])
    experiment_id = int(ref_sig.iloc[int(index)]['Experiment_id'] - 1)
    channel = int(ref_sig.iloc[int(index)]['Channel']-1)

    # Decode needed for the subsequent data loading and extraction of the correspondent file name

    if isinstance(data_dir, bytes):
        data_dir = data_dir.decode()
    if isinstance(participant_id, bytes):
        participant_id = participant_id.decode()
    if (int(participant_id) < 10):
        file_name = str('s0' + str(participant_id))
    else:
        file_name = str('s' + str(participant_id))

    # File path creation - data loading, considering only 'data' part, the correspondent channel and from 3rd second
    # (with a sampling rate of 128, corresponds from value 384)

    file_dat = str(data_dir + '/' + str(file_name) + '.dat')
    data = cPickle.load(open(file_dat, 'rb'), encoding='latin1')['data'][experiment_id, channel, 384:]

    return data


def bandpassfilter(sig, lowcut, highcut, fs, order=5):

    # Pass band filtering for a digital signal

    b, a = scipy.signal.butter(order, [lowcut, highcut], btype='bandpass', fs=fs, analog=False)
    y = scipy.signal.filtfilt(b, a, sig, axis=0)

    return y


def load_and_normalize_and_filter_data(index, data_dir):

    # Load data (in time-window) and divide the signal in its main 4 bands

    data = load_data(index, data_dir)
    filtered_data = np.zeros([data.shape[0], data.shape[1], 4])
    for i in range(data.shape[0]):
        filtered_data[i, :, 0] = bandpassfilter(data[i, :], 4.0, 8.0, 128)
        filtered_data[i, :, 1] = bandpassfilter(data[i, :], 8.0, 14.0, 128)
        filtered_data[i, :, 2] = bandpassfilter(data[i, :], 14.0, 30.0, 128)
        filtered_data[i, :, 3] = bandpassfilter(data[i, :], 30.0, 45.0, 128)

    # Scale data to have mean 0 and variance 1

    data_scaled = np.zeros(filtered_data.shape)
    for i in range(filtered_data.shape[2]):
        for j in range(filtered_data.shape[0]):
            data_scaled[j, :, i] = scale(filtered_data[j, :, i])

    return data_scaled.astype(np.float32)


def load_and_normalize_and_filter_signal(index, data_dir):

    # Load data (entire signal for one channel) and divide the signal in its main 4 bands

    data = load_signal(index, data_dir)
    filtered_data = np.zeros([data.shape[0], 4])
    filtered_data[:, 0] = bandpassfilter(data, 4.0, 8.0, 150)
    filtered_data[:, 1] = bandpassfilter(data, 8.0, 14.0, 150)
    filtered_data[:, 2] = bandpassfilter(data, 14.0, 30.0, 150)
    filtered_data[:, 3] = bandpassfilter(data, 30.0, 45.0, 150)

    # Scale data to have mean 0 and variance 1

    data_scaled = np.zeros(filtered_data.shape)
    for i in range(filtered_data.shape[1]):
        data_scaled[:, i] = scale(filtered_data[:, i])

    # Reshape in order to have a matrix with 60 row (60 seconds of signal) and 128 column (one column for each sample
    # of the second)

    data_scaled = data_scaled.reshape((60, 128, 4))

    return data_scaled.astype(np.float32)


def load_PCA(index, data_dir, components=32):

    # Load data

    data = load_and_normalize_and_filter_data(index, data_dir)

    # Extraction of the first 'components' principal components of the matrices

    pc_data = np.zeros([data.shape[0], components, data.shape[2]])
    for i in range(data.shape[2]):
        pca = PCA(n_components=components)
        pc_data[:, :, i] = pca.fit_transform(data[:, :, i])
    return pc_data.astype(np.float32)


def load_KPCA(index, data_dir, components=32, ker='rbf'):

    # Load data

    data = load_and_normalize_and_filter_data(index, data_dir)

    # Extraction of the first 'components' principal components of the matrices with a Kernel Principal Component Analysis

    kpc_data = np.zeros([data.shape[0], components, data.shape[2]])
    for i in range(4):
        kpca = KernelPCA(n_components=components, kernel=ker)
        kpc_data[:, :, i] = kpca.fit_transform(data[:, :, i])
    return kpc_data.astype(np.float32)


def load_ICA(index, data_dir, components=32, iter=500, toll=0.0001, alg='deflation'):

    # Load data

    data = load_and_normalize_and_filter_data(index, data_dir)

    # Extraction of the first 'components' independent components of the matrices with the Independent Component Analysis

    ICA_data = np.zeros([data.shape[0], components, data.shape[2]])
    for i in range(data.shape[2]):
        ica = FastICA(n_components=components, max_iter=iter, tol=toll, algorithm=alg)
        ICA_data[:, :, i] = ica.fit_transform(data[:, :, i])
    return ICA_data.astype(np.float32)


def load_PCC(index, data_dir):

    # Load data

    data = load_and_normalize_and_filter_data(index, data_dir)

    # Calculation of the Pearson Correlation Coefficients for the channels

    coeffs = np.zeros([data.shape[0], data.shape[0], data.shape[2]])
    for i in range(4):
        coeffs[:, :, i] = np.corrcoef(data[:, :, i])
    return coeffs.astype(np.float32)


def load_SC(index, data_dir):

    # Load data

    data = load_and_normalize_and_filter_data(index, data_dir)

    # Calculation of the four main statistical characteristics for the channels:
    # mean, variance, kurtosis, skewness

    data_SC = np.zeros([data.shape[0], 4, data.shape[2]])
    for i in range(data.shape[2]):
        for j in range(data.shape[0]):
            sc = scipy.stats.describe(data[j, :, i], axis=0)
            data_SC[j, :, i] = np.array([sc.mean, sc.variance, sc.kurtosis, sc.skewness]).T
    return data_SC.astype(np.float32)


def load_FFT(index, data_dir):

    # Load data

    data = load_and_normalize_and_filter_data(index, data_dir)

    # Extraction of the coefficient of the real Fast Fourier Transformation

    data_FFT = np.empty(shape=[data.shape[0], 513, data.shape[2]])
    for j in range(data.shape[2]):
        for i in range(data.shape[0]):
            yf = rfft(data[i, :, j])
            data_FFT[i, :, j] = np.abs(yf)
    return data_FFT.astype(np.float32)


def create_dataset(label, ref, function, input_size, batch_size, shuffle):
    # Creation of the dataset of type data - label

    # Extraction of data indexes of from the dataframe and labels (depending on the label names passed in input)

    data_indexes = list(ref.index)
    for i in range(len(data_indexes)):
        data_indexes[i] = str(data_indexes[i])
    labels = ref[label]

    # Creation of the dataset with indexes and label

    dataset = tf.data.Dataset.from_tensor_slices((data_indexes, labels))

    # Application of the function passed in input to every data index (from the index, data is extracted and if necessary
    # a feature is extracted with 'function'

    py_func = lambda index, label: (tf.numpy_function(function, [index, data_dir], np.float32), label)
    dataset = dataset.map(py_func, num_parallel_calls=os.cpu_count())

    # Operations for shuffling and batching of the dataset

    if shuffle: dataset = dataset.shuffle(len(data_indexes))
    dataset = dataset.repeat()

    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=1)

    return dataset


def create_dataset_nolabel(ref, function, input_size, batch_size, shuffle):
    # Creation of the dataset of type data - data for autoencoders

    # Extraction of data indexes of from the dataframe and labels (depending on the label names passed in input)

    file_indexes = list(ref.index)
    for i in range(len(file_indexes)):
        file_indexes[i] = str(file_indexes[i])
    file_indexes2 = file_indexes

    # Creation of the dataset with indexes and label

    dataset = tf.data.Dataset.from_tensor_slices((file_indexes, file_indexes2))

    # Application of the function passed in input to every data index (from the index, data is extracted and if
    # necessary a feature is extracted with 'function' (equal for both the data present)

    py_func = lambda index, index2: (tf.numpy_function(function, [index, data_dir], np.float32), tf.numpy_function(function, [index2, data_dir], np.float32))
    dataset = dataset.map(py_func, num_parallel_calls=os.cpu_count())

    if shuffle: dataset = dataset.shuffle(len(file_indexes))
    dataset = dataset.repeat()

    # Operations for shuffling and batching of the dataset

    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=1)

    return dataset


def print_evaluation(title, history, model, test_dataset, label):

    # Extract labels of test set, predict them with the model

    test_labels = test_ref_wind[label].values
    test_preds = model.predict(test_dataset, steps=test_steps)[:test_labels.shape[0]].squeeze()
    test_est_classes = (test_preds > 0.5).astype(int)

    # Determine performance scores

    accuracy = accuracy_score(test_labels, test_est_classes, normalize=True)
    precision, recall, fscore, _ = precision_recall_fscore_support(test_labels, test_est_classes, average='macro')

    print('PERFORMANCES ON TEST SET:')
    print('Accuracy: {:.2f}%'.format(accuracy * 100))
    print('Precision: {:.2f}%'.format(precision * 100))
    print('Recall: {:.2f}%'.format(recall * 100))
    print('Fscore: {:.2f}%'.format(fscore * 100))

    # Plot of loss-accuracy and ROC

    fig, axs = plt.subplots(2, 2)
    fig.suptitle('Loss, accuracy and ROC')
    # Plot loss
    axs[0, 0].plot(history.history['loss'], label='Train loss')
    axs[0, 0].plot(history.history['val_loss'], label='Val loss')
    axs[0, 0].legend()
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].set_title('Loss')
    # Plot accuracy
    axs[1, 0].plot(history.history['accuracy'], label='Train accuracy')
    axs[1, 0].plot(history.history['val_accuracy'], label='Val accuracy')
    axs[1, 0].legend()
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Accuracy')
    axs[1, 0].set_title('Accuracy')
    if len(label)==1:
        fpr, tpr, _ = roc_curve(test_labels, test_est_classes)
        roc_auc = auc(fpr, tpr)
        # Plot ROC when only 1 label is present
        axs[0, 1].plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        axs[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axs[0, 1].set_xlabel('False Positive Rate')
        axs[0, 1].set_ylabel('True Positive Rate')
        axs[0, 1].set_title('ROC for {}'.format(label))
    else:
        for l in range(len(label)):
            fpr, tpr, _ = roc_curve(test_labels[:, l], test_est_classes[:, l])
            roc_auc = auc(fpr, tpr)
            # Plot ROC for each of the two labels
            axs[l, 1].plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
            axs[l, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axs[l, 1].set_xlabel('False Positive Rate')
            axs[l, 1].set_ylabel('True Positive Rate')
            axs[l, 1].set_title('ROC for {}'.format(label[l]))
    plt.savefig('images/{}_evaluation'.format(title))


def print_evaluation_signal(title, history, model, test_dataset, label):

    # Extract labels of test set, predict them with the model

    test_labels = test_ref_sig[label].values
    test_preds = model.predict(test_dataset, steps=test_steps)[:test_labels.shape[0]].squeeze()
    test_est_classes = (test_preds > 0.5).astype(int)

    # Determine performance scores

    accuracy = accuracy_score(test_labels, test_est_classes, normalize=True)
    precision, recall, fscore, _ = precision_recall_fscore_support(test_labels, test_est_classes, average='macro')

    print('PERFORMANCES ON TEST SET:')
    print('Accuracy: {:.2f}%'.format(accuracy * 100))
    print('Precision: {:.2f}%'.format(precision * 100))
    print('Recall: {:.2f}%'.format(recall * 100))
    print('Fscore: {:.2f}%'.format(fscore * 100))

    # Plot of loss-accuracy and ROC

    fig, axs = plt.subplots(2, 2)
    fig.suptitle('Loss, accuracy and ROC')
    # Plot loss
    axs[0, 0].plot(history.history['loss'], label='Train loss')
    axs[0, 0].plot(history.history['val_loss'], label='Val loss')
    axs[0, 0].legend()
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].set_title('Loss')
    # Plot accuracy
    axs[1, 0].plot(history.history['accuracy'], label='Train accuracy')
    axs[1, 0].plot(history.history['val_accuracy'], label='Val accuracy')
    axs[1, 0].legend()
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Accuracy')
    axs[1, 0].set_title('Accuracy')
    if len(label)==1:
        fpr, tpr, _ = roc_curve(test_labels, test_est_classes)
        roc_auc = auc(fpr, tpr)
        # Plot ROC when only 1 label is present
        axs[0, 1].plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        axs[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axs[0, 1].set_xlabel('False Positive Rate')
        axs[0, 1].set_ylabel('True Positive Rate')
        axs[0, 1].set_title('ROC for {}'.format(label))
    else:
        for l in range(len(label)):
            fpr, tpr, _ = roc_curve(test_labels[:, l], test_est_classes[:, l])
            roc_auc = auc(fpr, tpr)
            # Plot ROC for each of the two labels
            axs[l, 1].plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
            axs[l, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axs[l, 1].set_xlabel('False Positive Rate')
            axs[l, 1].set_ylabel('True Positive Rate')
            axs[l, 1].set_title('ROC for {}'.format(label[l]))
    plt.savefig('images/{}_evaluation'.format(title))