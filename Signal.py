from Global_signal import *
from Functions import *
from Models import *
from Models_2labels import *
from ResNet import *

# Signal dataset

input_shape = (60, 128, 4)
early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Arousal

train_dataset = create_dataset('Arousal_label', train_ref_sig, function=load_and_normalize_and_filter_signal,
                               input_size=input_shape, batch_size=batch_size, shuffle=False)
val_dataset = create_dataset('Arousal_label', val_ref_sig, function=load_and_normalize_and_filter_signal,
                             input_size=input_shape, batch_size=batch_size, shuffle=False)
test_dataset = create_dataset('Arousal_label', test_ref_sig, function=load_and_normalize_and_filter_signal,
                             input_size=input_shape, batch_size=batch_size, shuffle=False)

print(train_dataset)
# ------ DNN for signal

print("DNN for signal")
model = DNN(input_shape)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/signal_DNN.h5')
print_evaluation_signal('arousal/signal_DNN', history, model, test_dataset, ['Arousal_label'])

# ------- AE for signal

print("DAE for signal")
encoder, decoder = DAE(input_shape, code_size=512)

inp = tf.keras.Input(input_shape)
code = encoder(inp)
reconstruction = decoder(code)
autoencoder = tf.keras.Model(inputs=inp, outputs=reconstruction)
autoencoder.compile(optimizer="adam", loss="mse")

train_dataset_nolabel = create_dataset_nolabel(train_ref_sig, function=load_and_normalize_and_filter_signal,
                                               input_size=input_shape, batch_size=batch_size, shuffle=False)
val_dataset_nolabel = create_dataset_nolabel(val_ref_sig, function=load_and_normalize_and_filter_signal,
                                           input_size=input_shape, batch_size=batch_size, shuffle=False)
test_dataset_nolabel = create_dataset_nolabel(test_ref_sig, function=load_and_normalize_and_filter_signal,
                                           input_size=input_shape, batch_size=batch_size, shuffle=False)

history = autoencoder.fit(train_dataset_nolabel, epochs=num_epochs, steps_per_epoch=train_steps,
                validation_data=val_dataset_nolabel, validation_steps=val_steps, callbacks=[early_stop_callback])

autoencoder.save('models/signal_AE.h5')
reconstruction_mse, reconstruction_accuracy = autoencoder.evaluate(test_dataset_nolabel, verbose=0)
print("MSE:", reconstruction_mse)
print("Accuracy:", reconstruction_accuracy)

# autoencoder = tf.keras.models.load_model('models/signal_AE.h5')

model = DNN_model(autoencoder)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/signal_DNNAE.h5')
print_evaluation_signal('arousal/signal_DNNAE', history, model, test_dataset, ['Arousal_label'])

# ------ CNN for signal

print("CNN for signal")
model = CNN(input_shape)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/signal_CNN.h5')
print_evaluation_signal('arousal/signal_CNN', history, model, test_dataset, ['Arousal_label'])

# model = tf.keras.models.load_model('models/arousal/signal_CNN.h5')

model_DNN = DNN_model(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/arousal/signal_DNNCNN.h5')
print_evaluation_signal('arousal/signal_DNNCNN', history, model_DNN, test_dataset, ['Arousal_label'])

# ------ CNNd for signal

print("CNNd for signal")
model = CNNd(input_shape)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/signal_CNNd.h5')
print_evaluation_signal('arousal/signal_CNNd', history, model, test_dataset, ['Arousal_label'])

# model = tf.keras.models.load_model('models/arousal/signal_CNNd.h5')

model_DNN = DNN_model(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/arousal/signal_DNNCNNd.h5')
print_evaluation_signal('arousal/signal_DNNCNNd', history, model_DNN, test_dataset, ['Arousal_label'])

# ------ CNNl for signal

print("CNNl for signal")
model = CNNl(input_shape)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/signal_CNNl.h5')
print_evaluation_signal('arousal/signal_CNNl', history, model, test_dataset, ['Arousal_label'])

# model = tf.keras.models.load_model('models/arousal/signal_CNNl.h5')

model_DNN = DNN_model(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/arousal/signal_DNNCNNl.h5')
print_evaluation_signal('arousal/signal_DNNCNNl', history, model_DNN, test_dataset, ['Arousal_label'])

# ------ RNN for signal

print("RNN for signal")
model = RNN(input_shape)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/signal_RNN.h5')
print_evaluation_signal('arousal/signal_RNN', history, model, test_dataset, ['Arousal_label'])

# model = tf.keras.models.load_model('models/arousal/signal_RNN.h5')
#
# model_DNN = DNN_model(model)
# model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
# history = model_DNN.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
#                     validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])
#
# model_DNN.save('models/arousal/signal_DNNRNN.h5')
# print_evaluation_signal('arousal/signal_DNNRNN', history, model_DNN, test_dataset, ['Arousal_label'])

# ------ ResNet for signal

print("ResNet for signal")
model = ResNet(input_shape)
model.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/signal_ResNet.h5')
print_evaluation_signal('arousal/signal_ResNet', history, model, test_dataset, ['Arousal_label'])


# Valence

train_dataset = create_dataset('Valence_label', train_ref_sig, function=load_and_normalize_and_filter_signal,
                               input_size=input_shape, batch_size=batch_size, shuffle=False)
val_dataset = create_dataset('Valence_label', val_ref_sig, function=load_and_normalize_and_filter_signal,
                             input_size=input_shape, batch_size=batch_size, shuffle=False)
test_dataset = create_dataset('Valence_label', test_ref_sig, function=load_and_normalize_and_filter_signal,
                             input_size=input_shape, batch_size=batch_size, shuffle=False)

# ------ DNN for signal

print("DNN for signal")
model = DNN(input_shape)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/signal_DNN.h5')
print_evaluation_signal('valence/signal_DNN', history, model, test_dataset, ['Valence_label'])

# ------- AE for signal

print("AE for signal")
autoencoder = tf.keras.models.load_model('models/signal_AE.h5')

model = DNN_model(autoencoder)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/signal_DNNAE.h5')
print_evaluation_signal('valence/signal_DNNAE', history, model, test_dataset, ['Valence_label'])

# ------ CNN for signal

print("CNN for signal")
model = CNN(input_shape)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/signal_CNN.h5')
print_evaluation_signal('valence/signal_CNN', history, model, test_dataset, ['Valence_label'])

# model = tf.keras.models.load_model('models/valence/signal_CNN.h5')

model_DNN = DNN_model(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/valence/signal_DNNCNN.h5')
print_evaluation_signal('valence/signal_DNNCNN', history, model_DNN, test_dataset, ['Valence_label'])

# ------ CNNd for signal

print("CNNd for signal")
model = CNNd(input_shape)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/signal_CNNd.h5')
print_evaluation_signal('valence/signal_CNNd', history, model, test_dataset, ['Valence_label'])

# model = tf.keras.models.load_model('models/valence/signal_CNNd.h5')

model_DNN = DNN_model(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/valence/signal_DNNCNNd.h5')
print_evaluation_signal('valence/signal_DNNCNNd', history, model_DNN, test_dataset, ['Valence_label'])

# ------ CNNl for signal

print("CNNl for signal")
model = CNNl(input_shape)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/signal_CNNl.h5')
print_evaluation_signal('valence/signal_CNNl', history, model, test_dataset, ['Valence_label'])

# model = tf.keras.models.load_model('models/valence/signal_CNNl.h5')

model_DNN = DNN_model(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model_DNN.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/valence/signal_DNNCNNl.h5')
print_evaluation_signal('valence/signal_DNNCNNl', history, model_DNN, test_dataset, ['Valence_label'])

# ------ RNN for signal

print("RNN for signal")
model = RNN(input_shape)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/signal_RNN.h5')
print_evaluation_signal('valence/signal_RNN', history, model, test_dataset, ['Valence_label'])

# model = tf.keras.models.load_model('models/valence/signal_RNN.h5')
#
# model_DNN = DNN_model(model)
# model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
# history = model_DNN.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
#                     validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])
#
# model_DNN.save('models/valence/signal_DNNRNN.h5')
# print_evaluation_signal('valence/signal_DNNRNN', history, model_DNN, test_dataset, ['Valence_label'])

# ------ ResNet for signal

model = ResNet(input_shape)
model.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/signal_ResNet.h5')
print_evaluation_signal('valence/signal_ResNet', history, model, test_dataset, ['Valence_label'])


# Both labels

train_dataset = create_dataset(['Valence_label','Arousal_label'], train_ref_sig, function=load_and_normalize_and_filter_signal,
                               input_size=input_shape, batch_size=batch_size, shuffle=False)
val_dataset = create_dataset(['Valence_label','Arousal_label'], val_ref_sig, function=load_and_normalize_and_filter_signal,
                             input_size=input_shape, batch_size=batch_size, shuffle=False)
test_dataset = create_dataset(['Valence_label','Arousal_label'], test_ref_sig, function=load_and_normalize_and_filter_signal,
                              input_size=input_shape, batch_size=batch_size, shuffle=False)

# ------ DNN for signal

print("DNN for signal")
model = DNN_2labels(input_shape)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/signal_DNN.h5')
print_evaluation_signal('both/signal_DNN', history, model, test_dataset, ['Valence_label','Arousal_label'])

# ------ DAE for signal

autoencoder = tf.keras.models.load_model('models/signal_AE.h5')

model = DNN_model(autoencoder)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/signal_DNNAE.h5')
print_evaluation_signal('both/signal_DNNAE', history, model, test_dataset, ['Valence_label','Arousal_label'])

# ------ CNN for signal

print("CNN for signal")
model = CNN_2labels(input_shape)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/signal_CNN.h5')
print_evaluation_signal('both/signal_CNN', history, model, test_dataset, ['Valence_label','Arousal_label'])

# model = tf.keras.models.load_model('models/both/signal_CNN.h5')

model_DNN = DNN_model_2labels(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/both/signal_DNNCNN.h5')
print_evaluation_signal('both/signal_DNNCNN', history, model_DNN, test_dataset, ['Valence_label','Arousal_label'])

# ------ CNNd for signal

print("CNNd for signal")
model = CNNd_2labels(input_shape)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/signal_CNNd.h5')
print_evaluation_signal('both/signal_CNNd', history, model, test_dataset, ['Valence_label','Arousal_label'])

# model = tf.keras.models.load_model('models/both/signal_CNNd.h5')

model_DNN = DNN_model_2labels(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/both/signal_DNNCNNd.h5')
print_evaluation_signal('both/signal_DNNCNNd', history, model_DNN, test_dataset, ['Valence_label','Arousal_label'])

# ------ CNNl for signal

print("CNNl for signal")
model = CNNl_2labels(input_shape)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/signal_CNNl.h5')
print_evaluation_signal('both/signal_CNNl', history, model, test_dataset, ['Valence_label','Arousal_label'])

# model = tf.keras.models.load_model('models/both/signal_CNNl.h5')

model_DNN = DNN_model_2labels(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/both/signal_DNNCNNl.h5')
print_evaluation_signal('both/signal_DNNCNNl', history, model_DNN, test_dataset, ['Valence_label','Arousal_label'])

# ------ RNN for signal

print("RNN for signal")
model = RNN_2labels(input_shape)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/signal_RNN.h5')
print_evaluation_signal('both/signal_RNN', history, model, test_dataset, ['Valence_label','Arousal_label'])

# model = tf.keras.models.load_model('models/both/signal_RNN.h5')
#
# model_DNN = DNN_model_2labels(model)
# model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# history = model_DNN.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
#                     validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])
#
# model_DNN.save('models/both/signal_DNNRNN.h5')
# print_evaluation_signal('both/signal_DNNRNN', history, model_DNN, test_dataset, ['Valence_label','Arousal_label'])

# ------ ResNet for signal

print("ResNet for signal")
model = ResNet_2(input_shape)
model.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/signal_ResNet.h5')
print_evaluation_signal('both/signal_ResNet', history, model, test_dataset, ['Valence_label','Arousal_label'])