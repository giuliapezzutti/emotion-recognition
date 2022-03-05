from Global import *
from Functions import *
from Models import *
from Models_2labels import *
from ResNet import *

# KPCA dataset

input_shape_KPCA = (32, 32, 4)
early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Arousal

train_dataset_KPCA = create_dataset('Arousal_label', train_ref_wind, function=load_KPCA,
                                   input_size=input_shape_KPCA, batch_size=batch_size, shuffle=False)
val_dataset_KPCA = create_dataset('Arousal_label', val_ref_wind, function=load_KPCA,
                                 input_size=input_shape_KPCA, batch_size=batch_size, shuffle=False)
test_dataset_KPCA = create_dataset('Arousal_label', test_ref_wind, function=load_KPCA,
                                 input_size=input_shape_KPCA, batch_size=batch_size, shuffle=False)

# ------ DNN for KPCA

print("DNN for KPCA")
model = DNN(input_shape_KPCA)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_KPCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_KPCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/KPCA_DNN.h5')
print_evaluation('arousal/KPCA_DNN', history, model, test_dataset_KPCA, ['Arousal_label'])

# ------- AE for KPCA

print("DAE for KPCA")
encoder, decoder = DAE(input_shape_KPCA, code_size=512)

inp = tf.keras.Input(input_shape_KPCA)
code = encoder(inp)
reconstruction = decoder(code)
autoencoder = tf.keras.Model(inputs=inp, outputs=reconstruction)
autoencoder.compile(optimizer="adam", loss="mse")

train_dataset_nolabel = create_dataset_nolabel(train_ref_wind, function=load_KPCA,
                                               input_size=input_shape_KPCA, batch_size=batch_size, shuffle=False)
val_dataset_nolabel = create_dataset_nolabel(val_ref_wind, function=load_KPCA,
                                           input_size=input_shape_KPCA, batch_size=batch_size, shuffle=False)
test_dataset_nolabel = create_dataset_nolabel(test_ref_wind, function=load_KPCA,
                                           input_size=input_shape_KPCA, batch_size=batch_size, shuffle=False)

history = autoencoder.fit(train_dataset_nolabel, epochs=num_epochs, steps_per_epoch=train_steps,
                          validation_data=val_dataset_nolabel, validation_steps=val_steps, callbacks=[early_stop_callback])

autoencoder.save('models/KPCA_AE.h5')
reconstruction_mse, reconstruction_accuracy = autoencoder.evaluate(test_dataset_nolabel, verbose=0)
print("MSE:", reconstruction_mse)
print("Accuracy:", reconstruction_accuracy)

# autoencoder = tf.keras.models.load_model('models/KPCA_AE.h5')

model = DNN_model(autoencoder)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset_KPCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_KPCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/KPCA_DNNAE.h5')
print_evaluation('arousal/KPCA_DNNAE', history, model, test_dataset_KPCA, ['Arousal_label'])

# ------ CNN for KPCA

print("CNN for KPCA")
model = CNN(input_shape_KPCA)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_KPCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_KPCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/KPCA_CNN.h5')
print_evaluation('arousal/KPCA_CNN', history, model, test_dataset_KPCA, ['Arousal_label'])

# model = tf.keras.models.load_model('models/arousal/KPCA_CNN.h5')

model_DNN = DNN_model(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset_KPCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_KPCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/arousal/KPCA_DNNCNN.h5')
print_evaluation('arousal/KPCA_DNNCNN', history, model_DNN, test_dataset_KPCA, ['Arousal_label'])

# ------ CNNd for KPCA

print("CNNd for KPCA")
model = CNNd(input_shape_KPCA)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_KPCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_KPCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/KPCA_CNNd.h5')
print_evaluation('arousal/KPCA_CNNd', history, model, test_dataset_KPCA, ['Arousal_label'])

# model = tf.keras.models.load_model('models/arousal/KPCA_CNNd.h5')

model_DNN = DNN_model(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset_KPCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_KPCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/arousal/KPCA_DNNCNNd.h5')
print_evaluation('arousal/KPCA_DNNCNNd', history, model_DNN, test_dataset_KPCA, ['Arousal_label'])

# ------ CNNl for KPCA

print("CNNl for KPCA")
model = CNNl(input_shape_KPCA)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_KPCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_KPCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/KPCA_CNNl.h5')
print_evaluation('arousal/KPCA_CNNl', history, model, test_dataset_KPCA, ['Arousal_label'])

# model = tf.keras.models.load_model('models/arousal/KPCA_CNNl.h5')

model_DNN = DNN_model(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset_KPCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_KPCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/arousal/KPCA_DNNCNNl.h5')
print_evaluation('arousal/KPCA_DNNCNNl', history, model_DNN, test_dataset_KPCA, ['Arousal_label'])

# ------ RNN for KPCA

print("RNN for KPCA")
model = RNN(input_shape_KPCA)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_KPCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_KPCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/KPCA_RNN.h5')
print_evaluation('arousal/KPCA_RNN', history, model, test_dataset_KPCA, ['Arousal_label'])

# model = tf.keras.models.load_model('models/arousal/KPCA_RNN.h5')
#
# model_DNN = DNN_model(model)
# model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
# history = model_DNN.fit(train_dataset_KPCA, epochs=num_epochs, steps_per_epoch=train_steps,
#                     validation_data=val_dataset_KPCA, validation_steps=val_steps, callbacks=[early_stop_callback])
#
# model_DNN.save('models/arousal/KPCA_DNNRNN.h5')
# print_evaluation('arousal/KPCA_DNNRNN', history, model_DNN, test_dataset_KPCA, ['Arousal_label'])

# ------ ResNet for KPCA

print("ResNet for KPCA")
model = ResNet(input_shape_KPCA)
model.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset_KPCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_KPCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/KPCA_ResNet.h5')
print_evaluation('arousal/KPCA_ResNet', history, model, test_dataset_KPCA, ['Arousal_label'])


# Valence

train_dataset_KPCA = create_dataset('Valence_label', train_ref_wind, function=load_KPCA,
                                   input_size=input_shape_KPCA, batch_size=batch_size, shuffle=False)
val_dataset_KPCA = create_dataset('Valence_label', val_ref_wind, function=load_KPCA,
                                 input_size=input_shape_KPCA, batch_size=batch_size, shuffle=False)
test_dataset_KPCA = create_dataset('Valence_label', test_ref_wind, function=load_KPCA,
                                 input_size=input_shape_KPCA, batch_size=batch_size, shuffle=False)

# ------ DNN for KPCA

print("DNN for KPCA")
model = DNN(input_shape_KPCA)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_KPCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_KPCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/KPCA_DNN.h5')
print_evaluation('valence/KPCA_DNN', history, model, test_dataset_KPCA, ['Valence_label'])

# ------- AE for KPCA

print("AE for KPCA")
autoencoder = tf.keras.models.load_model('models/KPCA_AE.h5')

model = DNN_model(autoencoder)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset_KPCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_KPCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/KPCA_DNNAE.h5')
print_evaluation('valence/KPCA_DNNAE', history, model, test_dataset_KPCA, ['Valence_label'])

# ------ CNN for KPCA

print("CNN for KPCA")
model = CNN(input_shape_KPCA)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_KPCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_KPCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/KPCA_CNN.h5')
print_evaluation('valence/KPCA_CNN', history, model, test_dataset_KPCA, ['Valence_label'])

# model = tf.keras.models.load_model('models/valence/KPCA_CNN.h5')

model_DNN = DNN_model(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset_KPCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_KPCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/valence/KPCA_DNNCNN.h5')
print_evaluation('valence/KPCA_DNNCNN', history, model_DNN, test_dataset_KPCA, ['Valence_label'])

# ------ CNNd for KPCA

print("CNNd for KPCA")
model = CNNd(input_shape_KPCA)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_KPCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_KPCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/KPCA_CNNd.h5')
print_evaluation('valence/KPCA_CNNd', history, model, test_dataset_KPCA, ['Valence_label'])

# model = tf.keras.models.load_model('models/valence/KPCA_CNNd.h5')

model_DNN = DNN_model(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset_KPCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_KPCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/valence/KPCA_DNNCNNd.h5')
print_evaluation('valence/KPCA_DNNCNNd', history, model_DNN, test_dataset_KPCA, ['Valence_label'])

# ------ CNNl for KPCA

print("CNNl for KPCA")
model = CNNl(input_shape_KPCA)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_KPCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_KPCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/KPCA_CNNl.h5')
print_evaluation('valence/KPCA_CNNl', history, model, test_dataset_KPCA, ['Valence_label'])

# model = tf.keras.models.load_model('models/valence/KPCA_CNNl.h5')

model_DNN = DNN_model(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset_KPCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_KPCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/valence/KPCA_DNNCNNl.h5')
print_evaluation('valence/KPCA_DNNCNNl', history, model_DNN, test_dataset_KPCA, ['Valence_label'])

# ------ RNN for KPCA

print("RNN for KPCA")
model = RNN(input_shape_KPCA)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_KPCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_KPCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/KPCA_RNN.h5')
print_evaluation('valence/KPCA_RNN', history, model, test_dataset_KPCA, ['Valence_label'])

# model = tf.keras.models.load_model('models/valence/KPCA_RNN.h5')
#
# model_DNN = DNN_model(model)
# model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
# history = model_DNN.fit(train_dataset_KPCA, epochs=num_epochs, steps_per_epoch=train_steps,
#                     validation_data=val_dataset_KPCA, validation_steps=val_steps, callbacks=[early_stop_callback])
#
# model_DNN.save('models/valence/KPCA_DNNRNN.h5')
# print_evaluation('valence/KPCA_DNNRNN', history, model_DNN, test_dataset_KPCA, ['Valence_label'])

# ------ ResNet for KPCA

print("ResNet for KPCA")
model = ResNet(input_shape_KPCA)
model.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset_KPCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_KPCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/KPCA_ResNet.h5')
print_evaluation('valence/KPCA_ResNet', history, model, test_dataset_KPCA, ['Valence_label'])


# dataset for both labels

train_dataset_KPCA = create_dataset(['Valence_label','Arousal_label'], train_ref_wind, function=load_KPCA,
                                   input_size=input_shape_KPCA, batch_size=batch_size, shuffle=False)
val_dataset_KPCA = create_dataset(['Valence_label','Arousal_label'], val_ref_wind, function=load_KPCA,
                                 input_size=input_shape_KPCA, batch_size=batch_size, shuffle=False)
test_dataset_KPCA = create_dataset(['Valence_label','Arousal_label'], test_ref_wind, function=load_KPCA,
                                 input_size=input_shape_KPCA, batch_size=batch_size, shuffle=False)

# ------ DNN for KPCA 

print("DNN for KPCA")
model = DNN_2labels(input_shape_KPCA)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_KPCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_KPCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/KPCA_DNN.h5')
print_evaluation('both/KPCA_DNN', history, model, test_dataset_KPCA, ['Valence_label','Arousal_label'])

# ------ DAE for KPCA

autoencoder = tf.keras.models.load_model('models/KPCA_AE.h5')

model = DNN_model_2labels(autoencoder)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset_KPCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_KPCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/KPCA_DNNAE.h5')
print_evaluation('both/KPCA_DNNAE', history, model, test_dataset_KPCA, ['Valence_label','Arousal_label'])

# ------ CNN for KPCA 

print("CNN for KPCA")
model = CNN_2labels(input_shape_KPCA)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_KPCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_KPCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/KPCA_CNN.h5')
print_evaluation('both/KPCA_CNN', history, model, test_dataset_KPCA, ['Valence_label','Arousal_label'])

# model = tf.keras.models.load_model('models/both/KPCA_CNN.h5')

model_DNN = DNN_model_2labels(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset_KPCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_KPCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/both/KPCA_DNNCNN.h5')
print_evaluation('both/KPCA_DNNCNN', history, model_DNN, test_dataset_KPCA, ['Valence_label','Arousal_label'])

# ------ CNNd for KPCA 

print("CNNd for KPCA")
model = CNNd_2labels(input_shape_KPCA)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_KPCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_KPCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/KPCA_CNNd.h5')
print_evaluation('both/KPCA_CNNd', history, model, test_dataset_KPCA, ['Valence_label','Arousal_label'])

# model = tf.keras.models.load_model('models/both/KPCA_CNNd.h5')

model_DNN = DNN_model_2labels(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset_KPCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_KPCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/both/KPCA_DNNCNNd.h5')
print_evaluation('both/KPCA_DNNCNNd', history, model_DNN, test_dataset_KPCA, ['Valence_label','Arousal_label'])

# ------ CNNl for KPCA 

print("CNNl for KPCA")
model = CNNl_2labels(input_shape_KPCA)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_KPCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_KPCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/KPCA_CNNl.h5')
print_evaluation('both/KPCA_CNNl', history, model, test_dataset_KPCA, ['Valence_label','Arousal_label'])

# model = tf.keras.models.load_model('models/both/KPCA_CNNl.h5')

model_DNN = DNN_model_2labels(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model_DNN.fit(train_dataset_KPCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_KPCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/both/KPCA_DNNCNNl.h5')
print_evaluation('both/KPCA_DNNCNNl', history, model_DNN, test_dataset_KPCA, ['Valence_label','Arousal_label'])

# ------ RNN for KPCA data

print("RNN for KPCA")
model = RNN_2labels(input_shape_KPCA)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_KPCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_KPCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/KPCA_RNN.h5')
print_evaluation('both/KPCA_RNN', history, model, test_dataset_KPCA, ['Valence_label','Arousal_label'])

# model = tf.keras.models.load_model('models/both/KPCA_RNN.h5')
#
# model_DNN = DNN_model_2labels(model)
# model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
# history = model_DNN.fit(train_dataset_KPCA, epochs=num_epochs, steps_per_epoch=train_steps,
#                     validation_data=val_dataset_KPCA, validation_steps=val_steps, callbacks=[early_stop_callback])
#
# model_DNN.save('models/both/KPCA_DNNRNN.h5')
# print_evaluation('both/KPCA_DNNRNN', history, model_DNN, test_dataset_KPCA, ['Valence_label','Arousal_label'])

# ------ ResNet for KPCA

model = ResNet_2(input_shape_KPCA)
model.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset_KPCA, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_KPCA, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/KPCA_ResNet.h5')
print_evaluation('both/KPCA_ResNet', history, model, test_dataset_KPCA, ['Valence_label','Arousal_label'])