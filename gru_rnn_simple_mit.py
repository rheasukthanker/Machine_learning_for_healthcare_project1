import pandas as pd
import numpy as np
import keras
from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D, \
    concatenate
from sklearn.metrics import f1_score, accuracy_score
#read data
df_train = pd.read_csv("mitbih_train.csv", header=None)
df_train = df_train.sample(frac=1)
df_test = pd.read_csv("mitbih_test.csv", header=None)

Y = np.array(df_train[187].values).astype(np.int8)
X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

Y_test = np.array(df_test[187].values).astype(np.int8)
X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]
RNN_OUTPUT_UNITS = [64, 128]
#define model
def get_model():
    inp = Input(shape=(187, 1))
    current_layer = keras.layers.Masking(mask_value=0., input_shape=(187, 1), name='Masked')(inp)
    for i, size in enumerate(RNN_OUTPUT_UNITS):
        notLast = i + 1 < len(RNN_OUTPUT_UNITS)
        current_layer = keras.layers.GRU(size, return_sequences=notLast, dropout=0.2, name = 'GRU' + str(i+1))(current_layer)

    current_layer = keras.layers.Dense(64, name='Dense1', activation='relu')(current_layer)
    current_layer = keras.layers.Dense(16, name='Dense2', activation='relu')(current_layer)
    logits = keras.layers.Dense(5, name='Output', activation='softmax')(current_layer)
    model = models.Model(inputs=inp, outputs=logits)
    opt = optimizers.Adam(0.001)

    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    model.summary()
    return model

model = get_model()
file_path = "gru_simple_mitbih.h5"
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=2)
callbacks_list = [checkpoint, early, redonplat]  # early
#train model
model.fit(X, Y, epochs=1000, verbose=2, callbacks=callbacks_list, validation_split=0.1)
model.load_weights(file_path)
pred_test = model.predict(X_test)
#Predict on test
pred_test = np.argmax(pred_test, axis=-1)

f1 = f1_score(Y_test, pred_test,average="macro")

print("Test f1 score : %s "% f1)

acc = accuracy_score(Y_test, pred_test)

print("Test accuracy score : %s "% acc)
