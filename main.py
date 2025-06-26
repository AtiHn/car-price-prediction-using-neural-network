import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


cd = pd.read_csv("data/car data.csv")
#remove the Car_Name columns (I think the name of cars can't be useful to prediction)
cd.drop(cd.columns[0], axis=1, inplace=True)

#define the age of cars
cd['car age'] = 2025 - cd['Year']
cd.drop('Year', axis=1, inplace=True)

#Convert categorical to numeric
cd = pd.get_dummies(cd, drop_first=True)
##print(cd.info())

# x (Features) ---> Things that the model learns from
# y (Label) ---> What the model is supposed to predict
x = cd.drop('Selling_Price', axis=1)
y = cd['Selling_Price']

#Splitting data into training and testing
#20% test and 80% train
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)
# X_temp, x_test , y_temp, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# train_x, val_x, train_y, val_y = train_test_split(x_temp, y_temp, test_size=0.2, random_state=42)

# X_train
#Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(train_x)
X_test = scaler.transform(test_x)

#Neural network model structure
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1])),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

#Compile the model
model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mean_absolute_error']
)

#Model training
history = model.fit(
    X_train, train_y,
    validation_split=0.2,
    epochs=50,
    batch_size=64,
    verbose=1
)



#Drawing a diagram
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#Assumption: history is taken from prior model training
train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(train_loss, label='Train Loss', color='green')
plt.plot(val_loss, label='Validation Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.title('Training vs Validation Loss')
plt.show()


#How well does the model predict on data it has not seen before?
loss, mae = model.evaluate(X_test, test_y)
print(f"Mean Absolute Error on test data: {mae:.2f}")
