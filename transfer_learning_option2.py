import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, losses, activations, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D, \
    concatenate
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM

df_train = pd.read_csv("heartbeat\mitbih_train.csv", header=None)
df_train = df_train.sample(frac=1)
df_test = pd.read_csv("heartbeat\mitbih_test.csv", header=None)

Y = np.array(df_train[187].values).astype(np.int8)
X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

Y_test = np.array(df_test[187].values).astype(np.int8)
X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]
RNN_OUTPUT_UNITS = [64, 128]

opt = optimizers.Adam(0.001)
model = Sequential()
model.add(LSTM(32, return_sequences=True,
               input_shape=(187, 1)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32))  # return a single vector of dimension 32
model.add(Dense(5, activation='softmax'))

model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
model.summary()

file_path = "lstm_mitbih.h5"
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=2)
callbacks_list = [checkpoint, early, redonplat]  # early

model.fit(X, Y, epochs=3, verbose=2, callbacks=callbacks_list, validation_split=0.1, class_weight=class_weights)
model.load_weights(file_path)
pred_test = model.predict(X_test)
pred_test = np.argmax(pred_test, axis=-1)

f1 = f1_score(Y_test, pred_test,average="macro")

print("Test f1 score : %s "% f1)

acc = accuracy_score(Y_test, pred_test)

print("Test accuracy score : %s "% acc)

# Remove output layer from the model trained on mit dataset above
model_mit=model
print(len(model_mit.layers))  
model_mit.pop()
print(len(model_mit.layers))

# Add output layer(s) for the PTB Diagnostic ECG Database
model_mit.add(Dense(2, activation='softmax'))
model_mit.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
model_mit.summary()

df_1 = pd.read_csv("heartbeat\ptbdb_normal.csv", header=None)
df_2 = pd.read_csv("heartbeat\ptbdb_abnormal.csv", header=None)
df = pd.concat([df_1, df_2])

df_train, df_test = train_test_split(df, test_size=0.2, random_state=1337, stratify=df[187])


Y = np.array(df_train[187].values).astype(np.int8)
X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

Y_test = np.array(df_test[187].values).astype(np.int8)
X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]

#model_mit.load_weights("lstm_mitbih.h5", by_name=True)
model_mit.fit(X, Y, epochs=3, verbose=2, callbacks=callbacks_list, validation_split=0.1)

pred_test = model_mit.predict(X_test)
pred_test = np.argmax(pred_test, axis=-1)

f1 = f1_score(Y_test, pred_test,average="macro")
print("Test f1 score : %s "% f1)

acc = accuracy_score(Y_test, pred_test)
print("Test accuracy score : %s "% acc)