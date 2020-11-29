# -*- coding: utf-8 -*-
"""
This script trains vgg face model to classify race.
It uses dataset containing face images extracted 
by dlib.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import  Model
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from PIL import Image
import argparse
import os
layers = tf.keras.layers


class CustomCheckPointCallback(tf.keras.callbacks.Callback):
      def __init__(self, path, mode="accuracy", verbose = 0):
          assert mode in ["accuracy", "loss"]
          self.path = path
          self.mode = mode
          self.verbose = verbose
          if mode == "accuracy":
                self.best_accuracy = 0.0
          else:
                self.best_loss = float('inf')
      def on_epoch_end(self, logs={}):
          if self.mode == "accuracy":
                if logs["val_accuracy"] > self.best_accuracy:
                      if self.verbose != 0:
                            print("Accuracy impoved from: {:.4f} to {:.4f}".format(self.best_accuracy, logs["val_accuracy"]))
                      self.best_accuracy = logs["val_accuracy"]
                      self.model.save(self.path)
                            
                else:
                      return
          else:
                if logs["val_loss"] < self.best_loss:
                      if self.verbose != 0:
                            print("Loss decreased from: {:.4f} to {:.4f}".format(self.best_loss, logs["val_loss"]))
                      self.best_loss = logs["val_loss"]
                      self.model.save(self.path)
                return;
IMG_SIZE = (224, 224)
def get_model():
  # Convolution Features
  vgg_features = VGGFace(include_top=False, model="senet50", input_shape=(224, 224, 3), pooling='avg') # pooling: None, avg or max

  nb_class = 5
  last_layer = vgg_features.get_layer('global_average_pooling2d_16').output
  x = layers.Dense(128, activation="relu")(last_layer)
  x = layers.Dropout(0.25)(x)
  x = layers.Dense(256, activation="relu")(x)
  x = layers.Dropout(0.25)(x)
  x = layers.Dense(512, activation="relu")(x)
  x = layers.Dropout(0.25)(x)
  out = layers.Dense(nb_class, activation='softmax', name='classifier')(x)
  custom_vgg_model = Model(inputs=vgg_features.input, outputs=out)
  for layer in vgg_features.layers:
      layer.trainable = False
  return custom_vgg_model

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True,
                        help="Path to split UTK-Face dataset after preprocessing with dlib. \nIt should contain folders `train`, `test` and `valid`")
    parser.add_argument("-d", "--exp_dir", default="exp", help="Experiment folder, to save checkpoints!")
    parser.add_argument("-b", "--batch_size", default=64, help="Batch size used during training", type=int)
    parser.add_argument("-e", "--epochs", default=100, help="Number of epochs to train the model", type=int)
    parser.add_argument('--multi-gpu', dest='multi_gpu', action='store_true', default=False)
    parser.add_argument("-l", "--lr", default=1e-4, help="Learning rate", type=float)
    args = parser.parse_args()
    return args



def get_data_gens(args):

    train_dir = os.path.join(args.path, "train")
    valid_dir = os.path.join(args.path, "valid")
    test_dir = os.path.join(args.path, "test")
    train_image_generator = ImageDataGenerator(preprocessing_function = preprocess_input,
                                              rotation_range = 30,
                                              width_shift_range = 0.2,
                                              height_shift_range = 0.2,
                                              zoom_range = 0.2,
                                              shear_range = 0.2,
                                              horizontal_flip = True
                                              )

    validation_image_generator = ImageDataGenerator(preprocessing_function = preprocess_input)
    test_image_generator = ImageDataGenerator(preprocessing_function = preprocess_input)

    train_image_gen = train_image_generator.flow_from_directory(
      batch_size = args.batch_size,
      directory = train_dir,
      shuffle = True,
      target_size = IMG_SIZE,
      class_mode= "categorical")

    validation_image_gen = validation_image_generator.flow_from_directory(
      batch_size = args.batch_size,
      directory = valid_dir,
      shuffle = False,
      target_size = IMG_SIZE,
      class_mode= "categorical")
    test_image_gen = validation_image_generator.flow_from_directory(
      batch_size = args.batch_size,
      directory = test_dir,
      shuffle = False,
      target_size = IMG_SIZE,
      class_mode= "categorical")
    return {"train":train_image_gen, "valid":validation_image_gen, "test":test_image_gen}

def main(args):
    if not os.path.exists(args.exp_dir):
          os.mkdir(args.exp_dir)
    if args.multi_gpu:
      strategy = tf.distribute.MirroredStrategy()
      with strategy.scope():
        model = get_model()
    else:
      model = get_model()
    print(model.summary())
    data_gens = get_data_gens(args)
    model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(lr=args.lr), metrics="accuracy")
    checkpoint_filepath = os.path.join(args.exp_dir, 'checkpoint.h5')
    # model_checkpoint_callback = CustomCheckPointCallback(checkpoint_filepath)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_filepath,
      save_weights_only=True,
      monitor='val_acc',
      mode='max',
      save_best_only=True)

    model.fit(data_gens["train"], validation_data = data_gens["valid"], epochs=args.epochs, callbacks=[model_checkpoint_callback], workers=8)
    
    model.save_weights(os.path.join(args.exp_dir, "final-weights.h5"))
    print(model.evaluate(data_gens["test"]))
if __name__ == '__main__':
    args = get_args()
    
    main(args)
    
