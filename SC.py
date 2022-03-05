from Global import *
from Functions import *
from Models import *
from Models_2labels import *
from ResNet import *

# SC dataset

input_shape_SC = (32, 4, 4)
early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Arousal

train_dataset_SC = create_dataset('Arousal_label', train_ref_wind, function=load_SC,
                                   input_size=input_shape_SC, batch_size=batch_size, shuffle=False)
val_dataset_SC = create_dataset('Arousal_label', val_ref_wind, function=load_SC,
                                 input_size=input_shape_SC, batch_size=batch_size, shuffle=False)
test_dataset_SC = create_dataset('Arousal_label', test_ref_wind, function=load_SC,
                                 input_size=input_shape_SC, batch_size=batch_size, shuffle=False)

# ------ DNN for SC

print("DNN for SC")
model = DNN(input_shape_SC)
model.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset_SC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_SC, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/SC_DNN.h5')
print_evaluation('arousal/SC_DNN', history, model, test_dataset_SC, ['Arousal_label'])

# ------- AE for SC data

print ("AE for SC")
encoder, decoder = DAE(input_shape_SC, code_size=256)

inp = tf.keras.Input(input_shape_SC)
code = encoder(inp)
reconstruction = decoder(code)
autoencoder = tf.keras.Model(inputs=inp, outputs=reconstruction)
autoencoder.compile(optimizer="adamax", loss='mse')

train_dataset_nolabel = create_dataset_nolabel(train_ref_wind, function=load_SC,
                                               input_size=input_shape_SC, batch_size=batch_size, shuffle=False)
val_dataset_nolabel = create_dataset_nolabel(val_ref_wind, function=load_SC,
                                           input_size=input_shape_SC, batch_size=batch_size, shuffle=False)
test_dataset_nolabel = create_dataset_nolabel(test_ref_wind, function=load_SC,
                                           input_size=input_shape_SC, batch_size=batch_size, shuffle=False)

autoencoder.fit(train_dataset_nolabel, epochs=num_epochs, steps_per_epoch=train_steps,
                validation_data=val_dataset_nolabel, validation_steps=val_steps, callbacks=[early_stop_callback])

autoencoder.save('models/SC_AE.h5')
reconstruction_mse, reconstruction_accuracy = autoencoder.evaluate(test_dataset_nolabel, verbose=0)
print("MSE:", reconstruction_mse)
print("Accuracy:", reconstruction_accuracy)

# autoencoder = tf.keras.models.load_model('models/SC_AE.h5')

model = DNN_model(autoencoder)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset_SC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_SC, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/SC_DNNAE.h5')
print_evaluation('arousal/SC_DNNAE', history, model, test_dataset_SC, ['Arousal_label'])

# ------ CNN for SC

print("CNN for SC")
model = CNN(input_shape_SC)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_SC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_SC, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/SC_CNN.h5')
print_evaluation('arousal/SC_CNN', history, model, test_dataset_SC, ['Arousal_label'])

# model = tf.keras.models.load_model('models/arousal/SC_CNN.h5')

model_DNN = DNN_model(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset_SC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_SC, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/arousal/SC_DNNCNN.h5')
print_evaluation('arousal/SC_DNNCNN', history, model_DNN, test_dataset_SC, ['Arousal_label'])

# ------ CNNd for SC

print("CNNd for SC")
model = CNNd(input_shape_SC)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_SC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_SC, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/SC_CNNd.h5')
print_evaluation('arousal/SC_CNNd', history, model, test_dataset_SC, ['Arousal_label'])

# model = tf.keras.models.load_model('models/arousal/SC_CNNd.h5')

model_DNN = DNN_model(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset_SC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_SC, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/arousal/SC_DNNCNNd.h5')
print_evaluation('arousal/SC_DNNCNNd', history, model_DNN, test_dataset_SC, ['Arousal_label'])

# ------ CNNl for SC

print("CNNl for SC")
model = CNNl(input_shape_SC)
model.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset_SC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_SC, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/SC_CNNl.h5')
print_evaluation('arousal/SC_CNNl', history, model, test_dataset_SC, ['Arousal_label'])

# model = tf.keras.models.load_model('models/arousal/SC_CNNl.h5')

model_DNN = DNN_model(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset_SC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_SC, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/arousal/SC_DNNCNNl.h5')
print_evaluation('arousal/SC_DNNCNNl', history, model_DNN, test_dataset_SC, ['Arousal_label'])

# ------ RNN for SC

print("\nRNN for SC\n")
model = RNN(input_shape_SC)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_SC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_SC, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/SC_RNN.h5')
print_evaluation('arousal/SC_RNN', history, model, test_dataset_SC, ['Arousal_label'])

# model = tf.keras.models.load_model('models/arousal/SC_RCNN.h5')
#
# model_DNN = DNN_model(model)
# model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
# history = model_DNN.fit(train_dataset_SC, epochs=num_epochs, steps_per_epoch=train_steps,
#                     validation_data=val_dataset_SC, validation_steps=val_steps, callbacks=[early_stop_callback])
#
# model_DNN.save('models/arousal/SC_DNNRNN.h5')
# print_evaluation('arousal/SC_DNNRNN', history, model_DNN, test_dataset_SC, ['Arousal_label'])

# ------ ResNet for SC

print("ResNet for SC")
model = ResNet(input_shape_SC)
model.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset_SC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_SC, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/arousal/SC_ResNet.h5')
print_evaluation('arousal/SC_ResNet', history, model, test_dataset_SC, ['Arousal_label'])


# Valence

train_dataset_SC = create_dataset('Valence_label', train_ref_wind, function=load_SC,
                                   input_size=input_shape_SC, batch_size=batch_size, shuffle=False)
val_dataset_SC = create_dataset('Valence_label', val_ref_wind, function=load_SC,
                                 input_size=input_shape_SC, batch_size=batch_size, shuffle=False)
test_dataset_SC = create_dataset('Valence_label', test_ref_wind, function=load_SC,
                                 input_size=input_shape_SC, batch_size=batch_size, shuffle=False)

# ------ DNN for SC

print("DNN for SC")
model = DNN(input_shape_SC)
model.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset_SC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_SC, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/SC_DNN.h5')
print_evaluation('valence/SC_DNN', history, model, test_dataset_SC, ['Valence_label'])

# ------- AE for SC data

print("AE for SC")
autoencoder = tf.keras.models.load_model('models/SC_AE.h5')

model = DNN_model(autoencoder)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset_SC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_SC, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/SC_DNNAE.h5')
print_evaluation('valence/SC_DNNAE', history, model, test_dataset_SC, ['Valence_label'])

# ------ CNN for SC

print("CNN for SC")
model = CNN(input_shape_SC)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_SC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_SC, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/SC_CNN.h5')
print_evaluation('valence/SC_CNN', history, model, test_dataset_SC, ['Valence_label'])

# model = tf.keras.models.load_model('models/valence/SC_CNN.h5')

model_DNN = DNN_model(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset_SC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_SC, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/valence/SC_DNNCNN.h5')
print_evaluation('valence/SC_DNNCNN', history, model_DNN, test_dataset_SC, ['Valence_label'])

# ------ CNNd for SC

print("CNNd for SC")
model = CNNd(input_shape_SC)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_SC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_SC, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/SC_CNNd.h5')
print_evaluation('valence/SC_CNNd', history, model, test_dataset_SC, ['Valence_label'])

# model = tf.keras.models.load_model('models/valence/SC_CNNd.h5')

model_DNN = DNN_model(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset_SC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_SC, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/valence/SC_DNNCNNd.h5')
print_evaluation('valence/SC_DNNCNNd', history, model_DNN, test_dataset_SC, ['Valence_label'])

# ------ CNNl for SC

print("CNNl for SC")
model = CNNl(input_shape_SC)
model.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset_SC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_SC, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/SC_CNNl.h5')
print_evaluation('valence/SC_CNNl', history, model, test_dataset_SC, ['Valence_label'])

# model = tf.keras.models.load_model('models/valence/SC_CNNl.h5')

model_DNN = DNN_model(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset_SC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_SC, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/valence/SC_DNNCNNl.h5')
print_evaluation('valence/SC_DNNCNNl', history, model_DNN, test_dataset_SC, ['Valence_label'])

# ------ RNN for SC

print("\nRNN for SC\n")
model = RNN(input_shape_SC)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_SC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_SC, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/SC_RNN.h5')
print_evaluation('valence/SC_RNN', history, model, test_dataset_SC, ['Valence_label'])

# model = tf.keras.models.load_model('models/valence/SC_RCNN.h5')
#
# model_DNN = DNN_model(model)
# model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
# history = model_DNN.fit(train_dataset_SC, epochs=num_epochs, steps_per_epoch=train_steps,
#                     validation_data=val_dataset_SC, validation_steps=val_steps, callbacks=[early_stop_callback])
#
# model_DNN.save('models/valence/SC_DNNRNN.h5')
# print_evaluation('valence/SC_DNNRNN', history, model_DNN, test_dataset_SC, ['Valence_label'])

# ------ ResNet for SC

print("ResNet for SC")
model = ResNet(input_shape_SC)
model.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset_SC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_SC, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/valence/SC_ResNet.h5')
print_evaluation('valence/SC_ResNet', history, model, test_dataset_SC, ['Valence_label'])


# Both

train_dataset_SC = create_dataset(['Valence_label','Arousal_label'], train_ref_wind, function=load_SC,
                                   input_size=input_shape_SC, batch_size=batch_size, shuffle=False)
val_dataset_SC = create_dataset(['Valence_label','Arousal_label'], val_ref_wind, function=load_SC,
                                 input_size=input_shape_SC, batch_size=batch_size, shuffle=False)
test_dataset_SC = create_dataset(['Valence_label','Arousal_label'], test_ref_wind, function=load_SC,
                                 input_size=input_shape_SC, batch_size=batch_size, shuffle=False)

# ------ DNN for SC

print("DNN for SC")
model = DNN_2labels(input_shape_SC)
model.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset_SC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_SC, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/SC_DNN.h5')
print_evaluation('both/SC_DNN', history, model, test_dataset_SC, ['Valence_label','Arousal_label'])

# ------ DAE for SC

print("DAE for SC")
autoencoder = tf.keras.models.load_model('models/SC_AE.h5')

model = DNN_model_2labels(autoencoder)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset_SC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_SC, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/SC_DNNAE.h5')
print_evaluation('both/SC_DNNAE', history, model, test_dataset_SC, ['Valence_label','Arousal_label'])

# ------ CNN for SC

print("CNN for SC")
model = CNN_2labels(input_shape_SC)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_SC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_SC, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/SC_CNN.h5')
print_evaluation('both/SC_CNN', history, model, test_dataset_SC, ['Valence_label','Arousal_label'])

# model = tf.keras.models.load_model('models/both/SC_CNN.h5')

model_DNN = DNN_model_2labels(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset_SC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_SC, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/both/SC_DNNCNN.h5')
print_evaluation('both/SC_DNNCNN', history, model_DNN, test_dataset_SC, ['Valence_label','Arousal_label'])

# ------ CNNd for SC

print("CNNd for SC")
model = CNNd_2labels(input_shape_SC)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_SC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_SC, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/SC_CNNd.h5')
print_evaluation('both/SC_CNNd', history, model, test_dataset_SC, ['Valence_label','Arousal_label'])

# model = tf.keras.models.load_model('models/both/SC_CNNd.h5')

model_DNN = DNN_model_2labels(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset_SC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_SC, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/both/SC_DNNCNNd.h5')
print_evaluation('both/SC_DNNCNNd', history, model_DNN, test_dataset_SC, ['Valence_label','Arousal_label'])

# ------ CNNl for SC

print("CNNl for SC")
model = CNNl_2labels(input_shape_SC)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_SC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_SC, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/SC_CNNl.h5')
print_evaluation('both/SC_CNNl', history, model, test_dataset_SC, ['Valence_label','Arousal_label'])

# model = tf.keras.models.load_model('models/both/SC_CNNl.h5')

model_DNN = DNN_model_2labels(model)
model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_DNN.fit(train_dataset_SC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_SC, validation_steps=val_steps, callbacks=[early_stop_callback])

model_DNN.save('models/both/SC_DNNCNNl.h5')
print_evaluation('both/SC_DNNCNNl', history, model_DNN, test_dataset_SC, ['Valence_label','Arousal_label'])

# ------ RNN for SC data

print("\nRNN for SC\n")
model = RNN_2labels(input_shape_SC)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset_SC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_SC, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/SC_RNN.h5')
print_evaluation('both/SC_RNN', history, model, test_dataset_SC, ['Valence_label','Arousal_label'])

# model = tf.keras.models.load_model('models/both/SC_RNN.h5')
#
# model_DNN = DNN_model_2labels(model)
# model_DNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
# history = model_DNN.fit(train_dataset_SC, epochs=num_epochs, steps_per_epoch=train_steps,
#                     validation_data=val_dataset_SC, validation_steps=val_steps, callbacks=[early_stop_callback])
#
# model_DNN.save('models/both/SC_DNNRNN.h5')
# print_evaluation('both/SC_DNNRNN', history, model_DNN, test_dataset_SC, ['Valence_label','Arousal_label'])

# ------ ResNet for SC

print("ResNet for SC")
model = ResNet_2(input_shape_SC)
model.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset_SC, epochs=num_epochs, steps_per_epoch=train_steps,
                    validation_data=val_dataset_SC, validation_steps=val_steps, callbacks=[early_stop_callback])

model.save('models/both/SC_ResNet.h5')
print_evaluation('both/SC_ResNet', history, model, test_dataset_SC, ['Valence_label','Arousal_label'])
