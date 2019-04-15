import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv("data/train.csv")       # read training data
test = pd.read_csv("data/test.csv")         # for calculating accuracy later

# print(train.shape)

from scipy.ndimage import imread

''' Reading images into code '''
def load_images(filenames, default_path):
    images = []

    for filename in filenames:
        filepath = default_path + filename

        image = imread(filepath)
        images.append(image)

    images = np.array(images)

    return images
''' '''

X_train = load_images(train["filename"], "data/train/")     #importing images
X_test = load_images(test["filename"], "data/test/")
#print(X_test.shape)

y_train = train["target"].values
y_test = test["target"].values

'''
figure, axes = plt.subplots(nrows=1, ncols=5)
figure.set_size_inches(18, 4)

axes[0].imshow(X_train[0])
axes[1].imshow(X_train[1])
axes[2].imshow(X_train[2])
axes[3].imshow(X_train[3])
axes[4].imshow(X_train[4])
'''

from tqdm import tqdm
from scipy.misc import imresize

def resize_image(original_images, size):       # resize images into same 224*224
    resized_images = []

    for original_image in tqdm(original_images):
        resized_image = imresize(original_image, size)
        resized_images.append(resized_image)

    resized_images = np.array(resized_images)

    return resized_images

X_train_224 = resize_image(X_train, (224, 224))
X_test_224 = resize_image(X_test, (224, 224))


#!pip install h5py

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

''' #Simple Convolutional Neural Network
model = Sequential()

model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same',
                 kernel_initializer='he_uniform', bias_initializer='zeros', input_shape=(224, 224, 3)))
model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same',
                 kernel_initializer='he_uniform', bias_initializer='zeros'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same',
                 kernel_initializer='he_uniform', bias_initializer='zeros'))
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same',
                 kernel_initializer='he_uniform', bias_initializer='zeros'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(units=128, activation='relu',
                kernel_initializer='he_uniform', bias_initializer='zeros'))
model.add(Dense(units=1, activation='sigmoid',
                kernel_initializer='glorot_uniform', bias_initializer='zeros'))

from keras.optimizers import SGD

optimizer = SGD(lr=0.000001)

model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_224, y_train, epochs=10)
'''

model = Sequential()        # VGGNet 2016

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                 padding='same',trainable=False, input_shape=(224, 224, 3)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                 padding='same', trainable=False))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu',
                 padding='same', trainable=False))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu',
                 padding='same', trainable=False))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu',
                 padding='same', trainable=False))
model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu',
                 padding='same', trainable=False))
model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu',
                 padding='same', trainable=False))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu',
                 padding='same', trainable=False))
model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu',
                 padding='same', trainable=False))
model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu',
                 padding='same', trainable=False))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu',
                 padding='same', trainable=False))
model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu',
                 padding='same', trainable=False))
model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu',
                 padding='same', trainable=False))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Load Weight from pre-trained model
model.load_weights("vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")

model.add(Flatten())
model.add(Dense(units=128, activation='relu',
                kernel_initializer='he_uniform', bias_initializer='zeros'))
model.add(Dense(units=1, activation='sigmoid',
                kernel_initializer='glorot_uniform', bias_initializer='zeros'))

from keras.optimizers import SGD

optimizer = SGD(lr=1e-4, momentum=0.9)

model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_224, y_train, epochs=1)       # creat fit model

y_predict = model.predict(X_test_224)   # run prediction
y_predict = (y_predict >= 0.5).reshape(-1).astype('int')  # tranform into 0 or 1

accuracy = (y_predict == y_test).mean()
print("Accuracy = {0:.5f}".format(accuracy))

prediction = test           # make csv file for prediction
prediction["target"] = y_predict
prediction.to_csv("prediction.csv", index=False)
