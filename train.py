import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

dataset_path = "dataset_lying_csv"
sequence_length = 299999
num_features = 1
X_list = []
y_list = []


for i in sorted(os.listdir(os.path.join(dataset_path, "NORMAL"))):
	filepath = os.path.join(os.path.join(dataset_path, "NORMAL"), i)
	data = pd.read_csv(filepath, header = None)
	X_list.append(data.values)
	y_list.append(0)

normal_len = len(X_list)
print("Total number of samples in NORMAL - ", normal_len)

for i in sorted(os.listdir(os.path.join(dataset_path, "POTS"))):
	filepath = os.path.join(os.path.join(dataset_path, "POTS"), i)
	data = pd.read_csv(filepath, header = None)
	X_list.append(data.values)
	y_list.append(1)

print("Total number of samples in POTS - ", len(X_list) - normal_len)

print("Total number of samples in the dataset - ", len(X_list))


X_train, X_val, y_train, y_val = train_test_split(X_list, y_list, test_size = 0.2)

X_train = np.array(X_train, dtype = np.float32)
y_train = np.array(y_train, dtype = np.float32)
X_val = np.array(X_val, dtype = np.float32)
y_val = np.array(y_val, dtype = np.float32)

X_train = X_train.reshape(-1, sequence_length, num_features)
X_val = X_val.reshape(-1, sequence_length, num_features)

print(X_train.shape)

model = keras.Sequential([
    keras.layers.LSTM(64, input_shape = (sequence_length, num_features)),
    keras.layers.Dense(1, activation = 'sigmoid') 
])


model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
print(model.summary())
model.fit(X_train, y_train, epochs = 1, batch_size = 8, validation_data = (X_val, y_val))

loss, accuracy = model.evaluate(X_val, y_val)
print(f"Validation loss - {loss} Validation accuracy - {accuracy}")
