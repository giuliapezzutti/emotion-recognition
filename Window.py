from Global import *
from Functions import *
from Models import *
from Models_2labels import *
from ResNet import *

# Window data dataset

input_shape = (32, 1024, 4)
early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Arousal 

train_dataset = create_dataset('Arousal_label', train_ref_wind, function=load_and_normalize_and_filter_data,
                               input_size=input_shape, batch_size=batch_size, shuffle=False)
val_dataset = create_dataset('Arousal_label', val_ref_wind, function=load_and_normalize_and_filter_data,
                             input_size=input_shape, batch_size=batch_size, shuffle=False)
test_dataset = create_dataset('Arousal_label', test_ref_wind, function=load_and_normalize_and_filter_data,
                              input_size=input_shape, batch_size=batch_size, shuffle=False)

# ------ DNN for windows

print("DNN for window")
model = DNN(input_shape)
model.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/window_DNN.h5')
print_evaluation('arousal/window_DNN', history, model, test_dataset, ['Arousal_label'])

# ------- AE for window

print("DAE for window")
encoder, decoder = DAE(input_shape, code_size=512)

inp = tf.keras.Input(input_shape)
code = encoder(inp)
reconstruction = decoder(code)
autoencoder = tf.keras.Model(inputs=inp, outputs=reconstruction)
autoencoder.compile(optimizer="adam", loss="mse")

train_dataset_nolabel = create_dataset_nolabel(train_ref_wind, function=load_and_normalize_and_filter_data,
                                               input_size=input_shape, batch_size=batch_size, shuffle=False)
val_dataset_nolabel = create_dataset_nolabel(val_ref_wind, function=load_and_normalize_and_filter_data,
                                           input_size=input_shape, batch_size=batch_size, shuffle=False)
test_dataset_nolabel = create_dataset_nolabel(test_ref_wind, function=load_and_normalize_and_filter_data,
                                           input_size=input_shape, batch_size=batch_size, shuffle=False)

autoencoder.fit(train_dataset_nolabel, epochs=num_epochs, steps_per_epoch=train_steps,
                validation_data=train_dataset_nolabel, validation_steps=val_steps, callbacks=[early_stop_callback])

autoencoder.save('models/window_AE.h5')
reconstruction_mse, reconstruction_accuracy = autoencoder.evaluate(test_dataset_nolabel, verbose=0)
print("MSE:", reconstruction_mse)
print("Accuracy:", reconstruction_accuracy)

# autoencoder = tf.keras.models.load_model('models/data_AE.h5')

model = DNN_model(autoencoder)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/window_DNNAE.h5')
print_evaluation('arousal/window_DNNAE', history, model, test_dataset, ['Arousal_label'])

# ------ CNN for window

print("CNN for window")
model = CNN(input_shape)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/window_CNN.h5')
print_evaluation('arousal/window_CNN', history, model, test_dataset, ['Arousal_label'])

# model = tf.keras.models.load_model('models/arousal/data_CNN.h5')

model_DNN = DNN_model(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/arousal/window_DNNCNN.h5')
print_evaluation('arousal/window_DNNCNN', history, model_DNN, test_dataset, ['Arousal_label'])

# ------ CNNd for window

print("CNNd for window")
model = CNNd(input_shape)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/window_CNNd.h5')
print_evaluation('arousal/window_CNNd', history, model, test_dataset, ['Arousal_label'])

# model = tf.keras.models.load_model('models/arousal/data_CNNd.h5')

model_DNN = DNN_model(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/arousal/window_DNNCNNd.h5')
print_evaluation('arousal/window_DNNCNNd', history, model_DNN, test_dataset, ['Arousal_label'])

# ------ CNNl for window

print("CNNl for window")
model = CNNl(input_shape)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/window_CNNl.h5')
print_evaluation('arousal/window_CNNl', history, model, test_dataset, ['Arousal_label'])

# model = tf.keras.models.load_model('models/arousal/data_CNNl.h5')

model_DNN = DNN_model(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/arousal/window_DNNCNNl.h5')
print_evaluation('arousal/window_DNNCNNl', history, model_DNN, test_dataset, ['Arousal_label'])

# ------ RNN for window

print("RNN for window")
model = RNN(input_shape)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/window_RNN.h5')
print_evaluation('arousal/window_RNN', history, model, test_windowset, ['Arousal_label'])

# model = tf.keras.models.load_model('models/arousal/window_RNN.h5')
#
# model_DNN = DNN_model(model)
# model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
# history = model_DNN.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
#                     validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])
#
# model_DNN.save('models/arousal/window_DNNRNN.h5')
# print_evaluation('arousal/window_DNNRNN', history, model_DNN, test_dataset, ['Arousal_label'])

# ------ ResNet for window

print("ResNet for window")
model = ResNet(input_shape)
model.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/window_ResNet.h5')
print_evaluation('arousal/window_ResNet', history, model, test_dataset, ['Arousal_label'])


# Valence 

train_dataset = create_dataset('Valence_label', train_ref_wind, function=load_and_normalize_and_filter_data,
                               input_size=input_shape, batch_size=batch_size, shuffle=False)
val_dataset = create_dataset('Valence_label', val_ref_wind, function=load_and_normalize_and_filter_data,
                             input_size=input_shape, batch_size=batch_size, shuffle=False)
test_dataset = create_dataset('Valence_label', test_ref_wind, function=load_and_normalize_and_filter_data,
                              input_size=input_shape, batch_size=batch_size, shuffle=False)

# ------ DNN for window

print("DNN for window")
model = DNN(input_shape)
model.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/window_DNN.h5')
print_evaluation('valence/window_DNN', history, model, test_windowset, ['Valence_label'])

# ------- AE for window

print("AE for window")
autoencoder = tf.keras.models.load_model('models/window_AE.h5')

model = DNN_model(autoencoder)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/window_DNNAE.h5')
print_evaluation('valence/window_DNNAE', history, model, test_dataset, ['Valence_label'])

# ------ CNN for window

print("CNN for window")
model = CNN(input_shape)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/window_CNN.h5')
print_evaluation('valence/window_CNN', history, model, test_dataset, ['Valence_label'])

# model = tf.keras.models.load_model('models/valence/data_CNN.h5')

model_DNN = DNN_model(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/valence/window_DNNCNN.h5')
print_evaluation('valence/window_DNNCNN', history, model_DNN, test_dataset, ['Valence_label'])

# ------ CNNd for window

print("CNNd for window")
model = CNNd(input_shape)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/window_CNNd.h5')
print_evaluation('valence/window_CNNd', history, model, test_dataset, ['Valence_label'])

# model = tf.keras.models.load_model('models/valence/data_CNNd.h5')

model_DNN = DNN_model(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/valence/window_DNNCNNd.h5')
print_evaluation('valence/window_DNNCNNd', history, model_DNN, test_dataset, ['Valence_label'])

# ------ CNNl for window

print("CNNl for window")
model = CNNl(input_shape)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/window_CNNl.h5')
print_evaluation('valence/window_CNNl', history, model, test_dataset, ['Valence_label'])

# model = tf.keras.models.load_model('models/valence/data_CNNl.h5')

model_DNN = DNN_model(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/valence/window_DNNCNNl.h5')
print_evaluation('valence/window_DNNCNNl', history, model_DNN, test_dataset, ['Valence_label'])

# ------ RNN for window

print("RNN for window")
model = RNN(input_shape)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/window_RNN.h5')
print_evaluation('valence/window_RNN', history, model, test_windowset, ['Valence_label'])

# model = tf.keras.models.load_model('models/valence/window_RNN.h5')
#
# model_DNN = DNN_model(model)
# model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
# history = model_DNN.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
#                     validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])
#
# model_DNN.save('models/valence/window_DNNRNN.h5')
# print_evaluation('valence/window_DNNRNN', history, model_DNN, test_dataset, ['Valence_label'])

# ------ ResNet for window

print("ResNet for window")
model = ResNet(input_shape)
model.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/window_ResNet.h5')
print_evaluation('valence/window_ResNet', history, model, test_dataset, ['Valence_label'])


# dataset for both labels

train_dataset = create_dataset(['Valence_label','Arousal_label'], train_ref_wind, function=load_and_normalize_and_filter_data,
                               input_size=input_shape, batch_size=batch_size, shuffle=False)
val_dataset = create_dataset(['Valence_label','Arousal_label'], val_ref_wind, function=load_and_normalize_and_filter_data,
                             input_size=input_shape, batch_size=batch_size, shuffle=False)
test_dataset = create_dataset(['Valence_label','Arousal_label'], test_ref_wind, function=load_and_normalize_and_filter_data,
                             input_size=input_shape, batch_size=batch_size, shuffle=False)

# ------ DNN for window

print("DNN for window")
model = DNN_2labels(input_shape)
model.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/window_ResNet.h5')
print_evaluation('both/window_ResNet', history, model, test_dataset, ['Valence_label','Arousal_label'])

# ------- DAE for window

print("AE for window")
autoencoder = tf.keras.models.load_model('models/window_AE.h5')

model = DNN_model_2labels(autoencoder)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/window_DNNAE.h5')
print_evaluation('both/window_DNNAE', history, model, test_dataset, ['Valence_label', 'Arousal_label'])

# ------ CNN for window

print("CNN for window")
model = CNN_2labels(input_shape)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/window_CNN.h5')
print_evaluation('both/window_CNN', history, model, test_dataset, ['Valence_label','Arousal_label'])

# model = tf.keras.models.load_model('models/both/data_CNN.h5')

model_DNN = DNN_model_2labels(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model_DNN.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/both/window_DNNCNN.h5')
print_evaluation('both/window_DNNCNN', history, model_DNN, test_dataset, ['Valence_label','Arousal_label'])

# ------ CNNd for window

print("CNNd for window")
model = CNNd_2labels(input_shape)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/window_CNNd.h5')
print_evaluation('both/window_CNNd', history, model, test_dataset, ['Valence_label','Arousal_label'])

# model = tf.keras.models.load_model('models/both/data_CNNd.h5')

model_DNN = DNN_model_2labels(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model_DNN.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/both/window_DNNCNNd.h5')
print_evaluation('both/window_DNNCNNd', history, model_DNN, test_dataset, ['Valence_label','Arousal_label'])

# ------ CNNl for window

print("CNNl for window")
model = CNNl_2labels(input_shape)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/window_CNNl.h5')
print_evaluation('both/window_CNNl', history, model, test_dataset, ['Valence_label', 'Arousal_label'])

# model = tf.keras.models.load_model('models/both/data_CNNl.h5')

model_DNN = DNN_model_2labels(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model_DNN.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/both/window_DNNCNNl.h5')
print_evaluation('both/window_DNNCNNl', history, model_DNN, test_dataset, ['Valence_label','Arousal_label'])

# ------ RNN for window

print("RNN for window")
model = RNN_2labels(input_shape)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/window_RNN.h5')
print_evaluation('both/window_RNN', history, model, test_dataset, ['Valence_label', 'Arousal_label'])

# model = tf.keras.models.load_model('models/both/window_RNN.h5')
#
# model_DNN = DNN_model_2labels(model)
# model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# history = model_DNN.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
#                     validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])
#
# model_DNN.save('models/both/window_DNNRNN.h5')
# print_evaluation('both/window_DNNRNN', history, model_DNN, test_dataset, ['Valence_label','Arousal_label'])

# ------- ResNet for window

print("ResNet for window")
model = ResNet_2(input_shape)
model.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/window_ResNet.h5')
print_evaluation('both/window_ResNet', history, model, test_dataset, ['Valence_label', 'Arousal_label'])
