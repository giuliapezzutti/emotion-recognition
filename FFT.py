from Global import *
from Functions import *
from Models import *
from Models_2labels import *
from ResNet import *

# FFT dataset

input_shape_FFT = (32, 513, 4)
early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Arousal

train_dataset_FFT = create_dataset('Arousal_label',train_ref_wind, function=load_FFT,
                                   input_size=input_shape_FFT, batch_size=batch_size, shuffle=False)
val_dataset_FFT = create_dataset('Arousal_label',val_ref_wind, function=load_FFT,
                                 input_size=input_shape_FFT, batch_size=batch_size,shuffle=False)
test_dataset_FFT = create_dataset('Arousal_label',test_ref_wind, function=load_FFT,
                                 input_size=input_shape_FFT, batch_size=batch_size,shuffle=False)

# ------ DNN for FFT

print("DNN for FFT")
model = DNN(input_shape_FFT)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset_FFT, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_FFT, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/FFT_DNN.h5')
print_evaluation('arousal/FFT_DNN', history, model, test_dataset_FFT, ['Arousal_label'])

# ------- AE for normal data

print("DAE for FFT")
encoder, decoder = DAE_FFT(input_shape_FFT, code_size=512)

inp = tf.keras.Input(input_shape_FFT)
code = encoder(inp)
reconstruction = decoder(code)
autoencoder = tf.keras.Model(inputs=inp, outputs=reconstruction)
autoencoder.compile(optimizer="adam", loss='mse')

train_dataset_nolabel = create_dataset_nolabel(train_ref_wind, function=load_FFT,
                                               input_size=input_shape_FFT, batch_size=batch_size, shuffle=False)
val_dataset_nolabel = create_dataset_nolabel(val_ref_wind, function=load_FFT,
                                             input_size=input_shape_FFT, batch_size=batch_size, shuffle=False)
test_dataset_nolabel = create_dataset_nolabel(test_ref_wind, function=load_FFT,
                                              input_size=input_shape_FFT, batch_size=batch_size, shuffle=False)

autoencoder.fit(train_dataset_nolabel, epochs=num_epochs, steps_per_epoch=train_steps,
                validation_data=val_dataset_nolabel, validation_steps=val_steps, callbacks=[early_stop_callback])

autoencoder.save('models/FFT_AE.h5')
reconstruction_mse, reconstruction_accuracy = autoencoder.evaluate(test_dataset_nolabel)
print("MSE:", reconstruction_mse)
print("Accuracy:", reconstruction_accuracy)

# autoencoder = tf.keras.models.load_model('models/FFT_AE.h5')

model = DNN_model(autoencoder)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset_FFT, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_FFT, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/FFT_DNNAE.h5')
print_evaluation('arousal/FFT_DNNAE', history, model, test_dataset_FFT, ['Arousal_label'])

# ------ CNN for FFT

print("CNN for FFT")
model = CNN(input_shape_FFT)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_FFT, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_FFT, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/FFT_CNN.h5')
print_evaluation('arousal/FFT_CNN', history, model, test_dataset_FFT, ['Arousal_label'])

# model = tf.keras.models.load_model('models/arousal/FFT_CNN.h5')

model_DNN = DNN_model(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset_FFT, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_FFT, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/arousal/FFT_DNNCNN.h5')
print_evaluation('arousal/FFT_DNNCNN', history, model_DNN, test_dataset_FFT, ['Arousal_label'])

# ------ CNNd for FFT

print("CNNd for FFT")
model = CNNd(input_shape_FFT)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_FFT, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_FFT, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/FFT_CNNd.h5')
print_evaluation('arousal/FFT_CNNd', history, model, test_dataset_FFT, ['Arousal_label'])

# model = tf.keras.models.load_model('models/arousal/FFT_CNNd.h5')

model_DNN = DNN_model(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset_FFT, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_FFT, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/arousal/FFT_DNNCNNd.h5')
print_evaluation('arousal/FFT_DNNCNNd', history, model_DNN, test_dataset_FFT, ['Arousal_label'])

# ------ CNNl for FFT

print("CNNl for FFT")
model = CNNl(input_shape_FFT)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_FFT, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_FFT, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/FFT_CNNl.h5')
print_evaluation('arousal/FFT_CNNl', history, model, test_dataset_FFT, ['Arousal_label'])

# model = tf.keras.models.load_model('models/arousal/FFT_CNNl.h5')

model_DNN = DNN_model(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset_FFT, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_FFT, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/arousal/FFT_DNNCNNl.h5')
print_evaluation('arousal/FFT_DNNCNNl', history, model_DNN, test_dataset_FFT, ['Arousal_label'])

# ------ RNN for PCA data

print("RNN for FFT data")
model = RNN(input_shape_FFT)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_FFT, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_FFT, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/FFT_RNN.h5')
print_evaluation('arousal/FFT_RNN', history, model, test_dataset_FFT, ['Arousal_label'])

# model = tf.keras.models.load_model('models/arousal/FFT_RNN.h5')
# 
# model_DNN = DNN_model(model)
# model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# 
# history = model_DNN.fit(train_dataset_FFT, epochs=num_epochs, steps_per_epoch=train_steps,
#                     validation_data=val_dataset_FFT, validation_steps=val_steps, callbacks=[early_stop_callback])
# 
# model_DNN.save('models/arousal/FFT_DNNRNN.h5')
# print_evaluation('arousal/FFT_DNNRNN', history, model_DNN, test_dataset_FFT, ['Arousal_label'])

# ------ ResNet for FFT

print("ResNet for FFT")
model = ResNet(input_shape_FFT)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset_FFT, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_FFT, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/FFT_ResNet.h5')
print_evaluation('arousal/FFT_ResNet', history, model, test_dataset_FFT, ['Arousal_label'])


# Valence

train_dataset_FFT = create_dataset('Valence_label',train_ref_wind, function=load_FFT,
                                   input_size=input_shape_FFT, batch_size=batch_size, shuffle=False)
val_dataset_FFT = create_dataset('Valence_label',val_ref_wind, function=load_FFT,
                                 input_size=input_shape_FFT, batch_size=batch_size,shuffle=False)
test_dataset_FFT = create_dataset('Valence_label',test_ref_wind, function=load_FFT,
                                 input_size=input_shape_FFT, batch_size=batch_size,shuffle=False)

# ------ DNN for FFT

print("DNN for FFT")
model = DNN(input_shape_FFT)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset_FFT, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_FFT, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/FFT_DNN.h5')
print_evaluation('valence/FFT_DNN', history, model, test_dataset_FFT, ['Valence_label'])

# ------- AE for normal data

print("DAE for FFT")
autoencoder = tf.keras.models.load_model('models/FFT_AE.h5')

model = DNN_model(autoencoder)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset_FFT, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_FFT, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/FFT_DNNAE.h5')
print_evaluation('valence/FFT_DNNAE', history, model, test_dataset_FFT, ['Valence_label'])

# ------ CNN for FFT

print("CNN for FFT")
model = CNN(input_shape_FFT)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_FFT, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_FFT, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/FFT_CNN.h5')
print_evaluation('valence/FFT_CNN', history, model, test_dataset_FFT, ['Valence_label'])

# model = tf.keras.models.load_model('models/valence/FFT_CNN.h5')

model_DNN = DNN_model(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset_FFT, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_FFT, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/valence/FFT_DNNCNN.h5')
print_evaluation('valence/FFT_DNNCNN', history, model_DNN, test_dataset_FFT, ['Valence_label'])

# ------ CNNd for FFT

print("CNNd for FFT")
model = CNNd(input_shape_FFT)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_FFT, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_FFT, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/FFT_CNNd.h5')
print_evaluation('valence/FFT_CNNd', history, model, test_dataset_FFT, ['Valence_label'])

# model = tf.keras.models.load_model('models/valence/FFT_CNNd.h5')

model_DNN = DNN_model(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset_FFT, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_FFT, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/valence/FFT_DNNCNNd.h5')
print_evaluation('valence/FFT_DNNCNNd', history, model_DNN, test_dataset_FFT, ['Valence_label'])

# ------ CNNl for FFT

print("CNNl for FFT")
model = CNNl(input_shape_FFT)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_FFT, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_FFT, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/FFT_CNNl.h5')
print_evaluation('valence/FFT_CNNl', history, model, test_dataset_FFT, ['Valence_label'])

# model = tf.keras.models.load_model('models/valence/FFT_CNNl.h5')

model_DNN = DNN_model(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset_FFT, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_FFT, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/valence/FFT_DNNCNNl.h5')
print_evaluation('valence/FFT_DNNCNNl', history, model_DNN, test_dataset_FFT, ['Valence_label'])

# ------ RNN for PCA data

print("RNN for FFT data")
model = RNN(input_shape_FFT)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_FFT, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_FFT, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/FFT_RNN.h5')
print_evaluation('valence/FFT_RNN', history, model, test_dataset_FFT, ['Valence_label'])

# model = tf.keras.models.load_model('models/valence/FFT_RNN.h5')
# 
# model_DNN = DNN_model(model)
# model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# 
# history = model_DNN.fit(train_dataset_FFT, epochs=num_epochs, steps_per_epoch=train_steps,
#                     validation_data=val_dataset_FFT, validation_steps=val_steps, callbacks=[early_stop_callback])
# 
# model_DNN.save('models/valence/FFT_DNNRNN.h5')
# print_evaluation('valence/FFT_DNNRNN', history, model_DNN, test_dataset_FFT, ['Valence_label'])

# ------ ResNet for FFT

print("ResNet for FFT")
model = ResNet(input_shape_FFT)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset_FFT, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_FFT, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/FFT_ResNet.h5')
print_evaluation('valence/FFT_ResNet', history, model, test_dataset_FFT, ['Valence_label'])


# Both

train_dataset_FFT = create_dataset(['Valence_label','Arousal_label'],train_ref_wind, function=load_FFT,
                                   input_size=input_shape_FFT, batch_size=batch_size, shuffle=False)
val_dataset_FFT = create_dataset(['Valence_label','Arousal_label'],val_ref_wind, function=load_FFT,
                                 input_size=input_shape_FFT, batch_size=batch_size, shuffle=False)
test_dataset_FFT = create_dataset(['Valence_label','Arousal_label'], test_ref_wind, function=load_FFT,
                                 input_size=input_shape_FFT, batch_size=batch_size, shuffle=False)

# ------ DNN for FFT

print("DNN for FFT")
model = DNN_2labels(input_shape_FFT)
model.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset_FFT, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_FFT, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/FFT_DNN.h5')
print_evaluation('both/FFT_DNN', history, model, test_dataset_FFT, ['Valence_label','Arousal_label'])

# ------ DAE for FFT

print("AE for FFT")
autoencoder = tf.keras.models.load_model('models/FFT_AE.h5')

model = DNN_model_2labels(autoencoder)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset_FFT, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_FFT, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/FFT_DNNAE.h5')
print_evaluation('both/FFT_DNNAE', history, model, test_dataset_FFT, ['Valence_label','Arousal_label'])

# ------ CNN for FFT

print("CNN for FFT")
model = CNN_2labels(input_shape_FFT)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_FFT, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_FFT, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/FFT_CNN.h5')
print_evaluation('both/FFT_CNN', history, model, test_dataset_FFT, ['Valence_label','Arousal_label'])

# model = tf.keras.models.load_model('models/both/FFT_CNN.h5')

model_DNN = DNN_model_2labels(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset_FFT, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_FFT, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/both/FFT_DNNCNN.h5')
print_evaluation('both/FFT_DNNCNN', history, model_DNN, test_dataset_FFT, ['Valence_label','Arousal_label'])

# ------ CNNd for FFT

print("CNNd for FFT")
model = CNNd_2labels(input_shape_FFT)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_FFT, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_FFT, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/FFT_CNNd.h5')
print_evaluation('both/FFT_CNNd', history, model, test_dataset_FFT, ['Valence_label','Arousal_label'])

# model = tf.keras.models.load_model('models/both/FFT_CNNd.h5')

model_DNN = DNN_model_2labels(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset_FFT, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_FFT, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/both/FFT_DNNCNNd.h5')
print_evaluation('both/FFT_DNNCNNd', history, model_DNN, test_dataset_FFT, ['Valence_label','Arousal_label'])

# ------ CNNl for FFT

print("CNNl for FFT")
model = CNNl_2labels(input_shape_FFT)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_FFT, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_FFT, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/FFT_CNNl.h5')
print_evaluation('both/FFT_CNNl', history, model, test_dataset_FFT, ['Valence_label','Arousal_label'])

# model = tf.keras.models.load_model('models/both/FFT_CNNl.h5')

model_DNN = DNN_model_2labels(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model_DNN.fit(train_dataset_FFT, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_FFT, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/both/FFT_DNNCNNl.h5')
print_evaluation('both/FFT_DNNCNNl', history, model_DNN, test_dataset_FFT, ['Valence_label','Arousal_label'])

# ------ RNN for FFT

print("RNN for FFT")
model = RNN_2labels(input_shape_FFT)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_FFT, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_FFT, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/FFT_RNN.h5')
print_evaluation('both/FFT_RNN', history, model, test_dataset_FFT, ['Valence_label','Arousal_label'])

# model = tf.keras.models.load_model('models/both/FFT_RNN.h5')
# 
# model_DNN = DNN_model_2labels(model)
# model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# 
# history = model_DNN.fit(train_dataset_FFT, epochs=num_epochs, steps_per_epoch=train_steps,
#                     validation_data=val_dataset_FFT, validation_steps=val_steps, callbacks=[early_stop_callback])
# 
# model_DNN.save('models/both/FFT_DNNRNN.h5')
# print_evaluation('both/FFT_DNNRNN', history, model_DNN, test_dataset_FFT, ['Valence_label','Arousal_label'])

# ------ ResNet for FFT

print("ResNet for FFT")
model = ResNet_2(input_shape_FFT)
model.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset_FFT, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_FFT, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/FFT_ResNet.h5')
print_evaluation('both/FFT_ResNet', history, model, test_dataset_FFT, ['Valence_label','Arousal_label'])