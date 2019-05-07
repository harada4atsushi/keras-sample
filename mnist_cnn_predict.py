from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import load_model
from keras import backend as K
import numpy as np

# input image dimensions
img_rows, img_cols = 28, 28

model = load_model('mnist_cnn_model.h5')

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


print('x_train.shape: ', x_train.shape)
x = x_train[:10]  # 先頭10件で予測させる
y = model.predict(x)

print('y_train[:10]:  ', y_train[:10])

# one-hotベクトルで結果が返るので、数値に変換する
y_classes = [np.argmax(v, axis=None, out=None) for v in y]
print('y_classes: ', y_classes)