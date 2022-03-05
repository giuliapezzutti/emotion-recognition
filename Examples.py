from pandas import DataFrame
from scipy.fft import rfftfreq
from Functions import *

index = 500
data_dir = 'data'

participant_id = int(ref_wind.iloc[int(index)]['Participant_id'])
experiment_id = int(ref_wind.iloc[int(index)]['Experiment_id'])
window_number = int(ref_wind.iloc[int(index)]['Window_number'])
arousal_label = int(ref_wind.iloc[int(index)]['Arousal_label'])
valence_label = int(ref_wind.iloc[int(index)]['Valence_label'])

print('Index: {}'.format(index))
print('Number of participant: {}'.format(participant_id))
print('Number of experiment: {}'.format(experiment_id))
print('Number of window: {}'.format(window_number))
print('Valence label: {}'.format(valence_label))
print('Arousal label: {}'.format(arousal_label))

data_nonorm = load_data(index, data_dir)
data = load_and_normalize_and_filter_data(index, data_dir)

data_PCA = load_PCA(index, data_dir)
data_KPCA = load_KPCA(index, data_dir)
data_ICA = load_ICA(index, data_dir)
data_PCC = load_PCC(index, data_dir)
data_SC = load_SC(index, data_dir)
data_FFT = load_FFT(index, data_dir)

filtered_data_alpha = bandpassfilter(data_nonorm[0, :], 1.0, 7.0, 150)
filtered_data_beta = bandpassfilter(data_nonorm[0, :], 8.0, 13.0, 150)
filtered_data_theta = bandpassfilter(data_nonorm[0, :], 14.0, 30.0, 150)
filtered_data_gamma = bandpassfilter(data_nonorm[0, :], 30.0, 45.0, 150)
reconstructed_data = np.sum([filtered_data_alpha, filtered_data_beta, filtered_data_theta, filtered_data_gamma], axis=0)

fig, axs = plt.subplots(3, 2)
axs[0, 0].plot(filtered_data_alpha)
axs[0, 0].set_title('Alpha 1-7Hz')
axs[0, 1].plot(filtered_data_beta, 'tab:orange')
axs[0, 1].set_title('Beta 8-13Hz')
axs[1, 0].plot(filtered_data_theta, 'tab:green')
axs[1, 0].set_title('Theta 14-30Hz')
axs[1, 1].plot(filtered_data_gamma, 'tab:red')
axs[1, 1].set_title('Gamma 30-45Hz')
axs[2, 0].plot(data_nonorm[0, :], 'tab:brown')
axs[2, 0].set_title('Initial Data')
axs[2, 1].plot(reconstructed_data, 'tab:pink')
axs[2, 1].set_title('Reconstructed Data')
plt.savefig('images/datafiltered.png')

fig, axs = plt.subplots(3, 2)
axs[0, 0].set_title('Data')
axs[0, 0].plot(filtered_data_alpha)
axs[0, 1].set_title('Data PCA')
axs[0, 1].plot(data_PCA[0, :, 0])
axs[1, 0].set_title('Data PCC')
axs[1, 0].plot(data_PCC[0, :, 0])
axs[1, 1].set_title('Data ICA')
axs[1, 1].plot(data_ICA[0, :, 0])
axs[2, 0].set_title('Data SC')
axs[2, 0].scatter([0,1,2,3], data_SC[0, :, 0])
axs[2, 1].set_title('Fourier Transformation')
axs[2, 1].plot(rfftfreq(128*8, 1/128), data_FFT[0,:,0])
fig.suptitle('Features extracted')
plt.savefig('images/features.png')
plt.show()

input_shape_SC = (32, 4, 4)

model = tf.keras.models.load_model('models/both/SC_DNNCNN.h5')

test = ref_wind.loc[[index], :]
test_dataset = create_dataset(['Valence_label','Arousal_label'], test, function=load_SC,
                              input_size=input_shape_SC, batch_size=batch_size, shuffle=False)

test_pred = model.predict(test_dataset, steps=1)[:1].squeeze()
test_est_class = (test_pred > 0.5).astype(int)

print('Estimated label: {}'.format(test_est_class))
