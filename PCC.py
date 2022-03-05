from Global import *
from Functions import *
from Models import *
from Models_2labels import *
from ResNet import *

# PCC dataset

input_shape_PCC = (32, 32, 4)
early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Arousal

train_dataset_PCC = create_dataset('Arousal_label', train_ref_wind, function=load_PCC,
                                   input_size=input_shape_PCC, batch_size=batch_size, shuffle=False)
val_dataset_PCC = create_dataset('Arousal_label', val_ref_wind, function=load_PCC,
                                 input_size=input_shape_PCC, batch_size=batch_size, shuffle=False)
test_dataset_PCC = create_dataset('Arousal_label', test_ref_wind, function=load_PCC,
                                 input_size=input_shape_PCC, batch_size=batch_size, shuffle=False)

# ------ DNN for PCC

print("DNN for PCC")
model = DNN(input_shape_PCC)
model.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset_PCC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCC, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/PCC_DNN.h5')
print_evaluation('arousal/PCC_DNN', history, model, test_dataset_PCC, ['Arousal_label'])

# ------- AE for PCC

print("AE for PCC")
encoder, decoder = DAE(input_shape_PCC, code_size=256)

inp = tf.keras.Input(input_shape_PCC)
code = encoder(inp)
reconstruction = decoder(code)
autoencoder = tf.keras.Model(inputs=inp, outputs=reconstruction)
autoencoder.compile(optimizer="adamax", loss='mse')

train_dataset_nolabel = create_dataset_nolabel(train_ref_wind, function=load_PCC,
                                               input_size=input_shape_PCC, batch_size=batch_size, shuffle=True)
val_dataset_nolabel = create_dataset_nolabel(val_ref_wind, function=load_PCC,
                                           input_size=input_shape_PCC, batch_size=batch_size, shuffle=True)
test_dataset_nolabel = create_dataset_nolabel(test_ref_wind, function=load_PCC,
                                           input_size=input_shape_PCC, batch_size=batch_size, shuffle=True)

autoencoder.fit(train_dataset_nolabel, epochs=num_epochs, steps_per_epoch=train_steps,
                validation_data=val_dataset_nolabel, validation_steps=val_steps, callbacks=[early_stop_callback])

autoencoder.save('models/PCC_AE.h5')
reconstruction_mse, reconstruction_accuracy = autoencoder.evaluate(test_dataset_nolabel, verbose=0)
print("MSE:", reconstruction_mse)
print("Accuracy:", reconstruction_accuracy)

# autoencoder = tf.keras.models.load_model('models/PCC_AE.h5')

model = DNN_model(autoencoder)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset_PCC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCC, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/PCC_DNNAE.h5')
print_evaluation('arousal/PCC_DNNAE', history, model, test_dataset_PCC, ['Arousal_label'])

# ------ CNN for PCC

print("CNN for PCC")
model = CNN(input_shape_PCC)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_PCC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCC, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/PCC_CNN.h5')
print_evaluation('arousal/PCC_CNN', history, model, test_dataset_PCC, ['Arousal_label'])

# model = tf.keras.models.load_model('models/arousal/PCC_CNN.h5')

model_DNN = DNN_model(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset_PCC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCC, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/arousal/PCC_DNNCNN.h5')
print_evaluation('arousal/PCC_DNNCNN', history, model_DNN, test_dataset_PCC, ['Arousal_label'])

# ------ CNNd for PCC

print("CNNd for PCC")
model = CNNd(input_shape_PCC)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_PCC, epochs=num_epochs, steps_per_epoch=train_steps, validation_data=val_dataset_PCC,
                    validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/PCC_CNNd.h5')
print_evaluation('arousal/PCC_CNNd', history, model, test_dataset_PCC, ['Arousal_label'])

# model = tf.keras.models.load_model('models/arousal/PCC_CNNd.h5')

model_DNN = DNN_model(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset_PCC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCC, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/arousal/PCC_DNNCNNd.h5')
print_evaluation('arousal/PCC_DNNCNNd', history, model_DNN, test_dataset_PCC, ['Arousal_label'])

# ------ CNNl for PCC

print("CNNl for PCC")
model = CNNl(input_shape_PCC)
model.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset_PCC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCC, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/CNNl_DNN.h5')
print_evaluation('arousal/CNNl_DNN', history, model, test_dataset_PCC, ['Arousal_label'])

# model = tf.keras.models.load_model('models/arousal/PCC_CNNl.h5')

model_DNN = DNN_model(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset_PCC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCC, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/arousal/PCC_DNNCNNl.h5')
print_evaluation('arousal/PCC_DNNCNNl', history, model_DNN, test_dataset_PCC, ['Arousal_label'])

# ------ RNN for PCC data

print("RNN for PCC")
model = RNN(input_shape_PCC)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_PCC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCC, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/PCC_RNN.h5')
print_evaluation('arousal/PCC_RNN', history, model, test_dataset_PCC, ['Arousal_label'])

# model = tf.keras.models.load_model('models/arousal/PCC_RNN.h5')
#
# model_DNN = DNN_model(model)
# model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
# history = model_DNN.fit(train_dataset_PCC, epochs=num_epochs, steps_per_epoch=train_steps,
#                     validation_data=val_dataset_PCC, validation_steps=val_steps, callbacks=[early_stop_callback])
#
# model_DNN.save('models/arousal/PCC_DNNRNN.h5')
# print_evaluation('arousal/PCC_DNNRNN', history, model_DNN, test_dataset_PCC, ['Arousal_label'])

# ------ ResNet for PCC

print("ResNet for PCC")
model = ResNet(input_shape_PCC)
model.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset_PCC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCC, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/PCC_ResNet.h5')
print_evaluation('arousal/PCC_ResNet', history, model, test_dataset_PCC, ['Arousal_label'])


# Valence

train_dataset_PCC = create_dataset('Valence_label', train_ref_wind, function=load_PCC,
                                   input_size=input_shape_PCC, batch_size=batch_size, shuffle=False)
val_dataset_PCC = create_dataset('Valence_label', val_ref_wind, function=load_PCC,
                                 input_size=input_shape_PCC, batch_size=batch_size, shuffle=False)
test_dataset_PCC = create_dataset('Valence_label', test_ref_wind, function=load_PCC,
                                 input_size=input_shape_PCC, batch_size=batch_size, shuffle=False)

# ------ DNN for PCC

print("DNN for PCC")
model = DNN(input_shape_PCC)
model.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset_PCC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCC, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/PCC_DNN.h5')
print_evaluation('valence/PCC_DNN', history, model, test_dataset_PCC, ['Valence_label'])

# ------- AE for PCC

print("AE for PCC")
autoencoder = tf.keras.models.load_model('models/PCC_AE.h5')

model = DNN_model(autoencoder)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset_PCC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCC, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/PCC_DNNAE.h5')
print_evaluation('valence/PCC_DNNAE', history, model, test_dataset_PCC, ['Valence_label'])

# ------ CNN for PCC

print("CNN for PCC")
model = CNN(input_shape_PCC)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_PCC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCC, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/PCC_CNN.h5')
print_evaluation('valence/PCC_CNN', history, model, test_dataset_PCC, ['Valence_label'])

# model = tf.keras.models.load_model('models/valence/PCC_CNN.h5')

model_DNN = DNN_model(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset_PCC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCC, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/valence/PCC_DNNCNN.h5')
print_evaluation('valence/PCC_DNNCNN', history, model_DNN, test_dataset_PCC, ['Valence_label'])

# ------ CNNd for PCC

print("CNNd for PCC")
model = CNNd(input_shape_PCC)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_PCC, epochs=num_epochs, steps_per_epoch=train_steps, validation_data=val_dataset_PCC,
                    validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/PCC_CNNd.h5')
print_evaluation('valence/PCC_CNNd', history, model, test_dataset_PCC, ['Valence_label'])

# model = tf.keras.models.load_model('models/valence/PCC_CNNd.h5')

model_DNN = DNN_model(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset_PCC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCC, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/valence/PCC_DNNCNNd.h5')
print_evaluation('valence/PCC_DNNCNNd', history, model_DNN, test_dataset_PCC, ['Valence_label'])

# ------ CNNl for PCC

print("CNNl for PCC")
model = CNNl(input_shape_PCC)
model.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset_PCC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCC, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/CNNl_DNN.h5')
print_evaluation('valence/CNNl_DNN', history, model, test_dataset_PCC, ['Valence_label'])

# model = tf.keras.models.load_model('models/valence/PCC_CNNl.h5')

model_DNN = DNN_model(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset_PCC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCC, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/valence/PCC_DNNCNNl.h5')
print_evaluation('valence/PCC_DNNCNNl', history, model_DNN, test_dataset_PCC, ['Valence_label'])

# ------ RNN for PCC data

print("RNN for PCC")
model = RNN(input_shape_PCC)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_PCC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCC, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/PCC_RNN.h5')
print_evaluation('valence/PCC_RNN', history, model, test_dataset_PCC, ['Valence_label'])

# model = tf.keras.models.load_model('models/valence/PCC_RNN.h5')
#
# model_DNN = DNN_model(model)
# model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
# history = model_DNN.fit(train_dataset_PCC, epochs=num_epochs, steps_per_epoch=train_steps,
#                     validation_data=val_dataset_PCC, validation_steps=val_steps, callbacks=[early_stop_callback])
#
# model_DNN.save('models/valence/PCC_DNNRNN.h5')
# print_evaluation('valence/PCC_DNNRNN', history, model_DNN, test_dataset_PCC, ['Valence_label'])

# ------ ResNet for PCC

print("ResNet for PCC")
model = ResNet(input_shape_PCC)
model.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset_PCC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCC, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/PCC_ResNet.h5')
print_evaluation('valence/PCC_ResNet', history, model, test_dataset_PCC, ['Valence_label'])


# Both labels

train_dataset_PCC = create_dataset(['Valence_label', 'Arousal_label'], train_ref_wind, function=load_PCC,
                                   input_size=input_shape_PCC, batch_size=batch_size, shuffle=False)
val_dataset_PCC = create_dataset(['Valence_label', 'Arousal_label'], val_ref_wind, function=load_PCC,
                                 input_size=input_shape_PCC, batch_size=batch_size, shuffle=False)
test_dataset_PCC = create_dataset(['Valence_label', 'Arousal_label'], test_ref_wind, function=load_PCC,
                                 input_size=input_shape_PCC, batch_size=batch_size, shuffle=False)

# ------ DNN for PCC

print("DNN for PCC")
model = DNN_2labels(input_shape_PCC)
model.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset_PCC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCC, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/PCC_DNN.h5')
print_evaluation('both/PCC_DNN', history, model, test_dataset_PCC, ['Valence_label','Arousal_label'])

# ------ DAE for PCC

print("AE for PCC")
autoencoder = tf.keras.models.load_model('models/PCC_AE.h5')

model = DNN_model_2labels(autoencoder)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset_PCC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCC, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/PCC_DNNAE.h5')
print_evaluation('both/PCC_DNNAE', history, model, test_dataset_PCC, ['Valence_label','Arousal_label'])

# ------ CNN for PCC data

print("CNN for PCC")
model = CNN_2labels(input_shape_PCC)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_PCC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCC, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/PCC_CNN.h5')
print_evaluation('both/PCC_CNN', history, model, test_dataset_PCC, ['Valence_label','Arousal_label'])

model = tf.keras.models.load_model('models/both/PCC_CNN.h5')

model_DNN = DNN_model_2labels(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model_DNN.fit(train_dataset_PCC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCC, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/both/PCC_DNNCNN.h5')
print_evaluation('both/PCC_DNNCNN', history, model_DNN, test_dataset_PCC, ['Valence_label','Arousal_label'])

# ------ CNNd for PCC data

print("CNNd for PCC")
model = CNNd_2labels(input_shape_PCC)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_PCC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCC, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/PCC_CNNd.h5')
print_evaluation('both/PCC_CNNd', history, model, test_dataset_PCC, ['Valence_label','Arousal_label'])

model = tf.keras.models.load_model('models/both/PCC_CNNd.h5')

model_DNN = DNN_model_2labels(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model_DNN.fit(train_dataset_PCC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCC, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/both/PCC_DNNCNNd.h5')
print_evaluation('both/PCC_DNNCNNd', history, model_DNN, test_dataset_PCC, ['Valence_label','Arousal_label'])

# ------ CNNl for PCC data

print("CNNl for PCC")
model = CNNl_2labels(input_shape_PCC)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_PCC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCC, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/PCC_CNNl.h5')
print_evaluation('both/PCC_CNNl', history, model, test_dataset_PCC, ['Valence_label','Arousal_label'])

# model = tf.keras.models.load_model('models/both/PCC_CNNl.h5')

model_DNN = DNN_model_2labels(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model_DNN.fit(train_dataset_PCC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCC, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/both/PCC_DNNCNNl.h5')
print_evaluation('both/PCC_DNNCNNl', history, model_DNN, test_dataset_PCC, ['Valence_label','Arousal_label'])

# ------ RNN for PCC data

print("RNN for PCC")
model = RNN_2labels(input_shape_PCC)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_PCC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCC, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/PCC_RNN.h5')
print_evaluation('both/PCC_RNN', history, model, test_dataset_PCC, ['Valence_label','Arousal_label'])

# model = tf.keras.models.load_model('models/both/PCC_RNN.h5')
#
# model_DNN = DNN_model_2labels(model)
# model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# history = model_DNN.fit(train_dataset_PCC, epochs=num_epochs, steps_per_epoch=train_steps,
#                     validation_data=val_dataset_PCC, validation_steps=val_steps, callbacks=[early_stop_callback])
#
# model_DNN.save('models/both/PCC_DNNRNN.h5')
# print_evaluation('both/PCC_DNNRNN', history, model_DNN, test_dataset_PCC, ['Valence_label','Arousal_label'])

# ------ ResNet for PCC

print("ResNet for PCC")
model = ResNet_2(input_shape_PCC)
model.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset_PCC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_PCC, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/PCC_ResNet.h5')
print_evaluation('both/PCC_ResNet', history, model, test_dataset_PCC, ['Valence_label','Arousal_label'])