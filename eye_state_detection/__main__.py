from tensorflow.keras.layers import Flatten, Dense, Input, GlobalAveragePooling2D, \
    GlobalMaxPooling2D, Activation, Conv2D, MaxPooling2D, BatchNormalization, \
    AveragePooling2D, Reshape, Permute, multiply
from keras_applications.imagenet_utils import _obtain_input_shape

"""
Declaimer
=========
The following code was adopted from https://github.com/rcmalli/keras-vggface/tree/master/keras_vggface
"""


# from tensorflow.keras.utils import layer_utils
from tensorflow.keras.utils import get_file
from tensorflow.keras import backend as K
# from keras_vggface import utils
# from keras.engine.topology import get_source_inputs
import warnings
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import tensorflow as tf
import os
from keras_vggface.vggface import VGGFace


def main():
    image_size = (224,224)
    vgg_model = VGGFace(model='vgg16', include_top=False, input_shape=(*image_size, 3))
    last_layer = vgg_model.get_layer('pool5').output
    x = Flatten(name='flatten')(last_layer)

    x = Dense(1, activation='sigmoid', name='output')(x)
    model = tf.keras.Model(inputs=vgg_model.inputs, outputs=x)
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(1e-4),
        metrics=['accuracy'],
    )
    for layer in vgg_model.layers:
        layer.trainable = False
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range = 20,
        rescale=1.0/255.0,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0.2,
        horizontal_flip = True,
        validation_split=0.2,
       
        
    )
    dataset_path = "dataset/eye-state"
    
    train_generator = datagen.flow_from_directory(dataset_path, batch_size = 32, subset ="training", class_mode="binary", target_size = image_size)
    valid_generator = datagen.flow_from_directory(dataset_path, batch_size = 32, subset ="validation", class_mode="binary",  target_size = image_size)
    # print(dir(train_generator))
    model.fit(
        train_generator,
        epochs=3,
        steps_per_epoch=len(train_generator),
        validation_data= valid_generator,
        validation_steps=len(valid_generator),

    )
    model.save("/content/gdrive/My Drive/Research-models/eye-state-detection-model.h5")

if __name__ == '__main__':
    main()
    