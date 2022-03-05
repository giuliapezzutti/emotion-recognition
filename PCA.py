from Global import *
from Functions import *
from Models import *
from Models_2labels import *
from ResNet import *

# PCA dataset

input_shape_PCA = (32, 32, 4)
early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Arousal

train_dataset_PCA = create_dataset('Arousal_label', train_ref_wind, function=load_PCA,
                                   input_size=input_shape_PCA, batch_size=batch_size, shuffle=False)
val_dataset_PCA = create_dataset('Arousal_label', val_ref_wind, function=load_PCA,
                                 input_size=input_shape_PCA, batch_size=batch_size, shuffle=False)
test_dataset_PCA = create_dataset('Arousal_label', test_ref_wind, function=load_PCA,
                                 input_size=input_shape_PCA, batch_size=batch_size, shuffle=False)

print(train_dataset_PCA)

# ------ DNN for PCA

print("DNN for PCA")
model = DNN(input_shape_PCA)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_PCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/PCA_DNN.h5')
print_evaluation('arousal/PCA_DNN', history, model, test_dataset_PCA, ['Arousal_label'])

# ------- AE for PCA

print("DAE for PCA")
encoder, decoder = DAE(input_shape_PCA, code_size=512)

inp = tf.keras.Input(input_shape_PCA)
code = encoder(inp)
reconstruction = decoder(code)
autoencoder = tf.keras.Model(inputs=inp, outputs=reconstruction)
autoencoder.compile(optimizer="adam", loss="mse")

train_dataset_nolabel = create_dataset_nolabel(train_ref_wind, function=load_PCA,
                                               input_size=input_shape_PCA, batch_size=batch_size, shuffle=False)
val_dataset_nolabel = create_dataset_nolabel(val_ref_wind, function=load_PCA,
                                           input_size=input_shape_PCA, batch_size=batch_size, shuffle=False)
test_dataset_nolabel = create_dataset_nolabel(test_ref_wind, function=load_PCA,
                                           input_size=input_shape_PCA, batch_size=batch_size, shuffle=False)

history = autoencoder.fit(train_dataset_nolabel, epochs=num_epochs, steps_per_epoch=train_steps,
                validation_data=val_dataset_nolabel, validation_steps=val_steps, callbacks=[early_stop_callback])

autoencoder.save('models/PCA_AE.h5')
reconstruction_mse, reconstruction_accuracy = autoencoder.evaluate(test_dataset_nolabel, verbose=0)
print("MSE:", reconstruction_mse)
print("Accuracy:", reconstruction_accuracy)

# autoencoder = tf.keras.models.load_model('models/PCA_AE.h5')

model = DNN_model(autoencoder)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset_PCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/PCA_DNNAE.h5')
print_evaluation('arousal/PCA_DNNAE', history, model, test_dataset_PCA, ['Arousal_label'])

# ------ CNN for PCA

print("CNN for PCA")
model = CNN(input_shape_PCA)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_PCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/PCA_CNN.h5')
print_evaluation('arousal/PCA_CNN', history, model, test_dataset_PCA, ['Arousal_label'])

# model = tf.keras.models.load_model('models/arousal/PCA_CNN.h5')

model_DNN = DNN_model(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset_PCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/arousal/PCA_DNNCNN.h5')
print_evaluation('arousal/PCA_DNNCNN', history, model_DNN, test_dataset_PCA, ['Arousal_label'])

# ------ CNNd for PCA

print("CNNd for PCA")
model = CNNd(input_shape_PCA)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_PCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/PCA_CNNd.h5')
print_evaluation('arousal/PCA_CNNd', history, model, test_dataset_PCA, ['Arousal_label'])

# model = tf.keras.models.load_model('models/arousal/PCA_CNNd.h5')

model_DNN = DNN_model(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset_PCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/arousal/PCA_DNNCNNd.h5')
print_evaluation('arousal/PCA_DNNCNNd', history, model_DNN, test_dataset_PCA, ['Arousal_label'])

# ------ CNNl for PCA

print("CNNl for PCA")
model = CNNl(input_shape_PCA)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_PCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/PCA_CNNl.h5')
print_evaluation('arousal/PCA_CNNl', history, model, test_dataset_PCA, ['Arousal_label'])

# model = tf.keras.models.load_model('models/arousal/PCA_CNNl.h5')

model_DNN = DNN_model(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset_PCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/arousal/PCA_DNNCNNl.h5')
print_evaluation('arousal/PCA_DNNCNNl', history, model_DNN, test_dataset_PCA, ['Arousal_label'])

# ------ RNN for PCA

print("RNN for PCA")
model = RNN(input_shape_PCA)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_PCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/PCA_RNN.h5')
print_evaluation('arousal/PCA_RNN', history, model, test_dataset_PCA, ['Arousal_label'])

# model = tf.keras.models.load_model('models/arousal/PCA_RNN.h5')
#
# model_DNN = DNN_model(model)
# model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
# history = model_DNN.fit(train_dataset_PCA, epochs=num_epochs, steps_per_epoch=train_steps,
#                     validation_data=val_dataset_PCA, validation_steps=val_steps, callbacks=[early_stop_callback])
#
# model_DNN.save('models/arousal/PCA_DNNRNN.h5')
# print_evaluation('arousal/PCA_DNNRNN', history, model_DNN, test_dataset_PCA, ['Arousal_label'])

# ------ ResNet for PCA

print("ResNet for PCA")
model = ResNet(input_shape_PCA)
model.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset_PCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/PCA_ResNet.h5')
print_evaluation('arousal/PCA_ResNet', history, model, test_dataset_PCA, ['Arousal_label'])


# Valence

train_dataset_PCA = create_dataset('Valence_label', train_ref_wind, function=load_PCA,
                                   input_size=input_shape_PCA, batch_size=batch_size, shuffle=False)
val_dataset_PCA = create_dataset('Valence_label', val_ref_wind, function=load_PCA,
                                 input_size=input_shape_PCA, batch_size=batch_size, shuffle=False)
test_dataset_PCA = create_dataset('Valence_label', test_ref_wind, function=load_PCA,
                                 input_size=input_shape_PCA, batch_size=batch_size, shuffle=False)

# ------ DNN for PCA

print("DNN for PCA")
model = DNN(input_shape_PCA)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_PCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/PCA_DNN.h5')
print_evaluation('valence/PCA_DNN', history, model, test_dataset_PCA, ['Valence_label'])

# ------- AE for PCA

print("AE for PCA")
autoencoder = tf.keras.models.load_model('models/PCA_AE.h5')

model = DNN_model(autoencoder)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset_PCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/PCA_DNNAE.h5')
print_evaluation('valence/PCA_DNNAE', history, model, test_dataset_PCA, ['Valence_label'])

# ------ CNN for PCA

print("CNN for PCA")
model = CNN(input_shape_PCA)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_PCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/PCA_CNN.h5')
print_evaluation('valence/PCA_CNN', history, model, test_dataset_PCA, ['Valence_label'])

# model = tf.keras.models.load_model('models/valence/PCA_CNN.h5')

model_DNN = DNN_model(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset_PCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/valence/PCA_DNNCNN.h5')
print_evaluation('valence/PCA_DNNCNN', history, model_DNN, test_dataset_PCA, ['Valence_label'])

# ------ CNNd for PCA

print("CNNd for PCA")
model = CNNd(input_shape_PCA)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_PCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/PCA_CNNd.h5')
print_evaluation('valence/PCA_CNNd', history, model, test_dataset_PCA, ['Valence_label'])

# model = tf.keras.models.load_model('models/valence/PCA_CNNd.h5')

model_DNN = DNN_model(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset_PCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/valence/PCA_DNNCNNd.h5')
print_evaluation('valence/PCA_DNNCNNd', history, model_DNN, test_dataset_PCA, ['Valence_label'])

# ------ CNNl for PCA

print("CNNl for PCA")
model = CNNl(input_shape_PCA)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_PCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/PCA_CNNl.h5')
print_evaluation('valence/PCA_CNNl', history, model, test_dataset_PCA, ['Valence_label'])

# model = tf.keras.models.load_model('models/valence/PCA_CNNl.h5')

model_DNN = DNN_model(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset_PCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/valence/PCA_DNNCNNl.h5')
print_evaluation('valence/PCA_DNNCNNl', history, model_DNN, test_dataset_PCA, ['Valence_label'])

# ------ RNN for PCA

print("RNN for PCA")
model = RNN(input_shape_PCA)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_PCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/PCA_RNN.h5')
print_evaluation('valence/PCA_RNN', history, model, test_dataset_PCA, ['Valence_label'])

# model = tf.keras.models.load_model('models/valence/PCA_RNN.h5')
#
# model_DNN = DNN_model(model)
# model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
# history = model_DNN.fit(train_dataset_PCA, epochs=num_epochs, steps_per_epoch=train_steps,
#                     validation_data=val_dataset_PCA, validation_steps=val_steps, callbacks=[early_stop_callback])
#
# model_DNN.save('models/valence/PCA_DNNRNN.h5')
# print_evaluation('valence/PCA_DNNRNN', history, model_DNN, test_dataset_PCA, ['Valence_label'])

# ------ ResNet for PCA

print("ResNet for PCA")
model = ResNet(input_shape_PCA)
model.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset_PCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/PCA_ResNet.h5')
print_evaluation('valence/PCA_ResNet', history, model, test_dataset_PCA, ['Valence_label'])


# Both

train_dataset_PCA = create_dataset(['Valence_label','Arousal_label'], train_ref_wind, function=load_PCA,
                                   input_size=input_shape_PCA, batch_size=batch_size, shuffle=False)
val_dataset_PCA = create_dataset(['Valence_label','Arousal_label'], val_ref_wind, function=load_PCA,
                                 input_size=input_shape_PCA, batch_size=batch_size, shuffle=False)
test_dataset_PCA = create_dataset(['Valence_label','Arousal_label'], test_ref_wind, function=load_PCA,
                                 input_size=input_shape_PCA, batch_size=batch_size, shuffle=False)

# ------ DNN for PCA

print("DNN for PCA")
model = DNN_2labels(input_shape_PCA)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_PCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/PCA_DNN.h5')
print_evaluation('both/PCA_DNN', history, model, test_dataset_PCA, ['Valence_label','Arousal_label'])

# ------ DAE for PCA

autoencoder = tf.keras.models.load_model('models/PCA_AE.h5')

model = DNN_model_2labels(autoencoder)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset_PCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/PCA_DNNAE.h5')
print_evaluation('both/PCA_DNNAE', history, model, test_dataset_PCA, ['Valence_label','Arousal_label'])

# ------ CNN for PCA

print("CNN for PCA")
model = CNN_2labels(input_shape_PCA)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_PCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/PCA_CNN.h5')
print_evaluation('both/PCA_CNN', history, model, test_dataset_PCA, ['Valence_label','Arousal_label'])

# model = tf.keras.models.load_model('models/both/PCA_CNN.h5')

model_DNN = DNN_model_2labels(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset_PCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/both/PCA_DNNCNN.h5')
print_evaluation('both/PCA_DNNCNN', history, model_DNN, test_dataset_PCA, ['Valence_label','Arousal_label'])

# ------ CNNd for PCA

print("CNNd for PCA")
model = CNNd_2labels(input_shape_PCA)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_PCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/PCA_CNNd.h5')
print_evaluation('both/PCA_CNNd', history, model, test_dataset_PCA, ['Valence_label','Arousal_label'])

# model = tf.keras.models.load_model('models/both/PCA_CNNd.h5')

model_DNN = DNN_model_2labels(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset_PCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/both/PCA_DNNCNNd.h5')
print_evaluation('both/PCA_DNNCNNd', history, model_DNN, test_dataset_PCA, ['Valence_label','Arousal_label'])

# ------ CNNl for PCA

print("CNNl for PCA")
model = CNNl_2labels(input_shape_PCA)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_PCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/PCA_CNNl.h5')
print_evaluation('both/PCA_CNNl', history, model, test_dataset_PCA, ['Valence_label','Arousal_label'])

# model = tf.keras.models.load_model('models/both/PCA_CNNl.h5')

model_DNN = DNN_model_2labels(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset_PCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/both/PCA_DNNCNNl.h5')
print_evaluation('both/PCA_DNNCNNl', history, model_DNN, test_dataset_PCA, ['Valence_label','Arousal_label'])

# ------ RNN for PCA data

print("RNN for PCA")
model = RNN_2labels(input_shape_PCA)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_PCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/PCA_RNN.h5')
print_evaluation('both/PCA_RNN', history, model, test_dataset_PCA, ['Valence_label','Arousal_label'])

# model = tf.keras.models.load_model('models/both/PCA_RNN.h5')
#
# model_DNN = DNN_model_2labels(model)
# model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
# history = model_DNN.fit(train_dataset_PCA, epochs=num_epochs, steps_per_epoch=train_steps,
#                     validation_data=val_dataset_PCA, validation_steps=val_steps, callbacks=[early_stop_callback])
#
# model_DNN.save('models/both/PCA_DNNRNN.h5')
# print_evaluation('both/PCA_DNNRNN', history, model_DNN, test_dataset_PCA, ['Valence_label','Arousal_label'])

# ------ ResNet for PCA

model = ResNet_2(input_shape_PCA)
model.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset_PCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/PCA_ResNet.h5')
print_evaluation('both/PCA_ResNet', history, model, test_dataset_PCA, ['Valence_label','Arousal_label'])