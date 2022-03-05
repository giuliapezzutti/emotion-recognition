from Global import *
from Functions import *
from Models import *
from Models_2labels import *
from ResNet import *

# ICA dataset

input_shape_ICA = (32, 32, 4)
early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Arousal

train_dataset_ICA = create_dataset('Arousal_label', train_ref_wind, function=load_ICA,
                                   input_size=input_shape_ICA, batch_size=batch_size, shuffle=False)
val_dataset_ICA = create_dataset('Arousal_label', val_ref_wind, function=load_ICA,
                                 input_size=input_shape_ICA, batch_size=batch_size, shuffle=False)
test_dataset_ICA = create_dataset('Arousal_label', test_ref_wind, function=load_ICA,
                                 input_size=input_shape_ICA, batch_size=batch_size, shuffle=False)

model = tf.keras.models.load_model('models/arousal/ICA_ResNet.h5')
label = ['Arousal_label']
test_labels = test_ref_wind[label].values
test_preds = model.predict(test_dataset_ICA, steps=test_steps)[:test_labels.shape[0]].squeeze()
test_est_classes = (test_preds > 0.5).astype(int)

# Determine performance scores

accuracy = accuracy_score(test_labels, test_est_classes, normalize=True)
precision, recall, fscore, _ = precision_recall_fscore_support(test_labels, test_est_classes, average='macro')

print('PERFORMANCES ON TEST SET:')
print('Accuracy: {:.2f}%'.format(accuracy * 100))
print('Precision: {:.2f}%'.format(precision * 100))
print('Recall: {:.2f}%'.format(recall * 100))
print('Fscore: {:.2f}%'.format(fscore * 100))

# ------ DNN for ICA

print("DNN for ICA")
model = DNN(input_shape_ICA)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_ICA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_ICA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/ICA_DNN.h5')
print_evaluation('arousal/ICA_DNN', history, model, test_dataset_ICA, ['Arousal_label'])

# ------- AE for ICA

print("DAE for ICA")
encoder, decoder = DAE(input_shape_ICA, code_size=256)

inp = tf.keras.Input(input_shape_ICA)
code = encoder(inp)
reconstruction = decoder(code)
autoencoder = tf.keras.Model(inputs=inp, outputs=reconstruction)
autoencoder.compile(optimizer="adamax", loss='mse')

train_dataset_nolabel = create_dataset_nolabel(train_ref_wind, function=load_ICA,
                                               input_size=input_shape_ICA, batch_size=batch_size, shuffle=False)
val_dataset_nolabel = create_dataset_nolabel(val_ref_wind, function=load_ICA,
                                           input_size=input_shape_ICA, batch_size=batch_size, shuffle=False)
test_dataset_nolabel = create_dataset_nolabel(test_ref_wind, function=load_ICA,
                                           input_size=input_shape_ICA, batch_size=batch_size, shuffle=False)

autoencoder.fit(train_dataset_nolabel, epochs=num_epochs, steps_per_epoch=train_steps,
                validation_data=val_dataset_nolabel, validation_steps=val_steps, callbacks=[early_stop_callback])

autoencoder.save('models/ICA_AE.h5')
reconstruction_mse, reconstruction_accuracy = autoencoder.evaluate(test_dataset_nolabel)
print("MSE:", reconstruction_mse)
print("Accuracy:", reconstruction_accuracy)

# autoencoder = tf.keras.models.load_model('models/ICA_AE.h5')

model = DNN_model(autoencoder)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset_ICA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_ICA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/ICA_DNNAE.h5')
print_evaluation('arousal/ICA_DNNAE', history, model, test_dataset_ICA, ['Arousal_label'])

# ------ CNN for ICA

print("CNN for ICA")
model = CNN(input_shape_ICA)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_ICA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_ICA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/ICA_CNN.h5')
print_evaluation('arousal/ICA_CNN', history, model, test_dataset_ICA, ['Arousal_label'])

# model = tf.keras.models.load_model('models/arousal/ICA_CNN.h5')

model_DNN = DNN_model(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset_ICA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_ICA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/ICA_DNNCNN.h5')
print_evaluation('arousal/ICA_DNNCNN', history, model_DNN, test_dataset_ICA, ['Arousal_label'])

# ------ CNNd for ICA

print("CNNd for ICA")
model = CNNd(input_shape_ICA)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_ICA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_ICA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/ICA_CNNd.h5')
print_evaluation('arousal/ICA_CNNd', history, model, test_dataset_ICA, ['Arousal_label'])

# model = tf.keras.models.load_model('models/arousal/ICA_CNNd.h5')

model_DNN = DNN_model(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset_ICA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_ICA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/ICA_DNNCNNd.h5')
print_evaluation('arousal/ICA_DNNCNNd', history, model_DNN, test_dataset_ICA, ['Arousal_label'])

# ------ CNNl for ICA

print("CNNl for ICA")
model = CNNd(input_shape_ICA)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_ICA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_ICA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/ICA_CNNl.h5')
print_evaluation('arousal/ICA_CNNl', history, model, test_dataset_ICA, ['Arousal_label'])

# model = tf.keras.models.load_model('models/arousal/ICA_CNNl.h5')

model_DNN = DNN_model(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset_ICA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_ICA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/ICA_DNNCNNl.h5')
print_evaluation('arousal/ICA_DNNCNNl', history, model_DNN, test_dataset_ICA, ['Arousal_label'])

# ------ RNN for ICA data

print("RNN for ICA")
model = RNN(input_shape_ICA)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_ICA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_ICA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/ICA_RNN.h5')
print_evaluation('arousal/ICA_RNN', history, model, test_dataset_ICA, ['Arousal_label'])

# model = tf.keras.models.load_model('models/arousal/ICA_RNN.h5')
# 
# model_DNN = DNN_model(model)
# model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# 
# history = model_DNN.fit(train_dataset_ICA, epochs=num_epochs, steps_per_epoch=train_steps,
#                     validation_data=val_dataset_ICA, validation_steps=val_steps, callbacks=[early_stop_callback])
# 
# model.save('models/arousal/ICA_DNNRNN.h5')
# print_evaluation('arousal/ICA_DNNRNN', history, model_DNN, test_dataset_ICA, ['Arousal_label'])

# ------ ResNet for ICA

print("RNN for ICA")
model = ResNet(input_shape_ICA)
model.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset_ICA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_ICA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/ICA_ResNet.h5')
print_evaluation('arousal/ICA_ResNet', history, model, test_dataset_ICA, ['Arousal_label'])


# Valence

train_dataset_ICA = create_dataset('Valence_label', train_ref_wind, function=load_ICA,
                                   input_size=input_shape_ICA, batch_size=batch_size, shuffle=False)
val_dataset_ICA = create_dataset('Valence_label', val_ref_wind, function=load_ICA,
                                 input_size=input_shape_ICA, batch_size=batch_size, shuffle=False)
test_dataset_ICA = create_dataset('Valence_label', test_ref_wind, function=load_ICA,
                                 input_size=input_shape_ICA, batch_size=batch_size, shuffle=False)

# ------ DNN for ICA

print("DNN for ICA")
model = DNN(input_shape_ICA)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_ICA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_ICA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/ICA_DNN.h5')
print_evaluation('valence/ICA_DNN', history, model, test_dataset_ICA, ['Valence_label'])

# ------- AE for ICA

print("DAE for ICA")
encoder, decoder = DAE(input_shape_ICA, code_size=256)

inp = tf.keras.Input(input_shape_ICA)
code = encoder(inp)
reconstruction = decoder(code)
autoencoder = tf.keras.Model(inputs=inp, outputs=reconstruction)
autoencoder.compile(optimizer="adamax", loss='mse')

train_dataset_nolabel = create_dataset_nolabel(train_ref_wind, function=load_ICA,
                                               input_size=input_shape_ICA, batch_size=batch_size, shuffle=False)
val_dataset_nolabel = create_dataset_nolabel(val_ref_wind, function=load_ICA,
                                           input_size=input_shape_ICA, batch_size=batch_size, shuffle=False)
test_dataset_nolabel = create_dataset_nolabel(test_ref_wind, function=load_ICA,
                                           input_size=input_shape_ICA, batch_size=batch_size, shuffle=False)

autoencoder.fit(train_dataset_nolabel, epochs=num_epochs, steps_per_epoch=train_steps,
                validation_data=val_dataset_nolabel, validation_steps=val_steps, callbacks=[early_stop_callback])

autoencoder.save('models/ICA_AE.h5')
reconstruction_mse, reconstruction_accuracy = autoencoder.evaluate(test_dataset_nolabel)
print("MSE:", reconstruction_mse)
print("Accuracy:", reconstruction_accuracy)

# autoencoder = tf.keras.models.load_model('models/ICA_AE.h5')

model = DNN_model(autoencoder)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset_ICA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_ICA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/ICA_DNNAE.h5')
print_evaluation('valence/ICA_DNNAE', history, model, test_dataset_ICA, ['Valence_label'])

# ------ CNN for ICA

print("CNN for ICA")
model = CNN(input_shape_ICA)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_ICA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_ICA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/ICA_CNN.h5')
print_evaluation('valence/ICA_CNN', history, model, test_dataset_ICA, ['Valence_label'])

# model = tf.keras.models.load_model('models/valence/ICA_CNN.h5')

model_DNN = DNN_model(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset_ICA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_ICA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/ICA_DNNCNN.h5')
print_evaluation('valence/ICA_DNNCNN', history, model_DNN, test_dataset_ICA, ['Valence_label'])

# ------ CNNd for ICA

print("CNNd for ICA")
model = CNNd(input_shape_ICA)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_ICA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_ICA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/ICA_CNNd.h5')
print_evaluation('valence/ICA_CNNd', history, model, test_dataset_ICA, ['Valence_label'])

# model = tf.keras.models.load_model('models/valence/ICA_CNNd.h5')

model_DNN = DNN_model(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset_ICA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_ICA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/ICA_DNNCNNd.h5')
print_evaluation('valence/ICA_DNNCNNd', history, model_DNN, test_dataset_ICA, ['Valence_label'])

# ------ CNNl for ICA

print("CNNl for ICA")
model = CNNd(input_shape_ICA)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_ICA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_ICA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/ICA_CNNl.h5')
print_evaluation('valence/ICA_CNNl', history, model, test_dataset_ICA, ['Valence_label'])

# model = tf.keras.models.load_model('models/valence/ICA_CNNl.h5')

model_DNN = DNN_model(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset_ICA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_ICA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/ICA_DNNCNNl.h5')
print_evaluation('valence/ICA_DNNCNNl', history, model_DNN, test_dataset_ICA, ['Valence_label'])

# ------ RNN for ICA data

print("RNN for ICA")
model = RNN(input_shape_ICA)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_ICA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_ICA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/ICA_RNN.h5')
print_evaluation('valence/ICA_RNN', history, model, test_dataset_ICA, ['Valence_label'])

# model = tf.keras.models.load_model('models/valence/ICA_RNN.h5')
# 
# model_DNN = DNN_model(model)
# model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# 
# history = model_DNN.fit(train_dataset_ICA, epochs=num_epochs, steps_per_epoch=train_steps,
#                     validation_data=val_dataset_ICA, validation_steps=val_steps, callbacks=[early_stop_callback])
# 
# model.save('models/valence/ICA_DNNRNN.h5')
# print_evaluation('valence/ICA_DNNRNN', history, model_DNN, test_dataset_ICA, ['Valence_label'])

# ------ ResNet for ICA

print("RNN for ICA")
model = ResNet(input_shape_ICA)
model.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset_ICA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_ICA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/ICA_ResNet.h5')
print_evaluation('valence/ICA_ResNet', history, model, test_dataset_ICA, ['Valence_label'])


# dataset for both labels

train_dataset_ICA = create_dataset(['Valence_label','Arousal_label'], train_ref_wind, function=load_ICA,
                                   input_size=input_shape_ICA, batch_size=batch_size, shuffle=False)
val_dataset_ICA = create_dataset(['Valence_label','Arousal_label'], val_ref_wind, function=load_ICA,
                                 input_size=input_shape_ICA, batch_size=batch_size,shuffle=False)
test_dataset_ICA = create_dataset(['Valence_label','Arousal_label'], test_ref_wind, function=load_ICA,
                                 input_size=input_shape_ICA, batch_size=batch_size, shuffle=False)

# ------ DNN for ICA

print("DNN for ICA")
model = DNN_2labels(input_shape_ICA)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_ICA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_ICA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/ICA_DNN.h5')
print_evaluation('both/ICA_DNN', history, model, test_dataset_ICA, ['Valence_label','Arousal_label'])

# ------ DAE for ICA

autoencoder = tf.keras.models.load_model('models/ICA_AE.h5')

model = DNN_model_2labels(autoencoder)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset_ICA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_ICA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/ICA_DNNAE.h5')
print_evaluation('both/ICA_DNNAE', history, model, test_dataset_ICA, ['Valence_label','Arousal_label'])

# ------ CNN for ICA data

print("CNN for ICA")
model = CNN_2labels(input_shape_ICA)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_ICA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_ICA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/ICA_CNN.h5')
print_evaluation('both/ICA_CNN', history, model, test_dataset_ICA, ['Valence_label', 'Arousal_label'])

# model = tf.keras.models.load_model('models/ICA_CNN.h5')

model_DNN = DNN_model_2labels(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset_ICA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_ICA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/ICA_DNNCNN.h5')
print_evaluation('both/ICA_DNNCNN', history, model_DNN, test_dataset_ICA, ['Valence_label','Arousal_label'])

# ------ CNNd for ICA data

print("CNNd for PCA")
model = CNNd_2labels(input_shape_ICA)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_ICA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_ICA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/ICA_CNNd.h5')
print_evaluation('both/ICA_CNNd', history, model, test_dataset_ICA, ['Valence_label', 'Arousal_label'])

# model = tf.keras.models.load_model('models/ICA_CNNd.h5')

model_DNN = DNN_model_2labels(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset_ICA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_ICA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/ICA_DNNCNNd.h5')
print_evaluation('both/ICA_DNNCNNd', history, model_DNN, test_dataset_ICA, ['Valence_label','Arousal_label'])

# ------ CNNl for ICA data

print("CNNl for PCA")
model = CNNl_2labels(input_shape_ICA)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_ICA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_ICA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/ICA_CNNl.h5')
print_evaluation('both/ICA_CNNl', history, model, test_dataset_ICA, ['Valence_label', 'Arousal_label'])

# model = tf.keras.models.load_model('models/ICA_CNNl.h5')

model_DNN = DNN_model_2labels(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset_ICA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_ICA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/ICA_DNNCNNl.h5')
print_evaluation('both/ICA_DNNCNNl', history, model_DNN, test_dataset_ICA, ['Valence_label','Arousal_label'])

# ------ RNN for ICA data

print("\nRNN for normal data\n")
model = RNN_2labels(input_shape_ICA)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_ICA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_ICA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/ICA_RNN.h5')
print_evaluation('both/ICA_RNN', history, model, test_dataset_ICA, ['Valence_label', 'Arousal_label'])

# model = tf.keras.models.load_model('models/ICA_RNN.h5')
#
# model_DNN = DNN_model_2labels(model)
# model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
# history = model_DNN.fit(train_dataset_ICA, epochs=num_epochs, steps_per_epoch=train_steps,
#                     validation_data=val_dataset_ICA, validation_steps=val_steps, callbacks=[early_stop_callback])
#
# model.save('models/both/ICA_DNNRNN.h5')
# print_evaluation('both/ICA_DNNRNN', history, model_DNN, test_dataset_ICA, ['Valence_label','Arousal_label'])

# ------ ResNet for PCA

model = ResNet_2(input_shape_ICA)
model.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset_ICA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_ICA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/ICA_ResNet.h5')
print_evaluation('both/ICA_ResNet', history, model, test_dataset_ICA, ['Valence_label','Arousal_label'])