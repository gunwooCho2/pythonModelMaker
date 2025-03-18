import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from dataset.h5py_util import H5pyLoad
from image.image_util import binary_image_unpack
import numpy as np


def __data_generator(temporary_dic):
    train = temporary_dic['train']
    label = temporary_dic['is_true']

    train = [binary_image_unpack(x, (512, 512)) for x in train]
    train = np.expand_dims(train, axis=-1)
    return train, label

def data_generator_wrapper(dataset):
    for temporary_dic in dataset.__get_item_all__():
        yield __data_generator(temporary_dic)


def main():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(512, 512, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    dataset = H5pyLoad("C:/Data/chess_board_detect/train_data/chess_dataset.h5", 128)
    output_types = (tf.float32, tf.int32)
    output_shapes = ((128, 512, 512, 1), (128, 1))

    train_dataset = tf.data.Dataset.from_generator(
        lambda: data_generator_wrapper(dataset),
        output_types=output_types,
        output_shapes=output_shapes
    )

    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    history = model.fit(train_dataset, epochs=10)


if __name__ == '__main__':
    main()