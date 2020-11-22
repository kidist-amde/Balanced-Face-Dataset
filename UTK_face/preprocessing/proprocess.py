# -*- coding: utf-8 -*-
"""
The following script is used to split the
utk face dataset into training, validation and test sets
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import shutil
import argparse


def get_race(path):
  """
  Labels
  The labels of each face image is embedded in the file name,
  formated like [age]_[gender]_[race]_[date&time].jpg

  [age] is an integer from 0 to 116, indicating the age
  [gender] is either 0 (male) or 1 (female)
  [race] is an integer from 0 to 4, denoting White, Black, Asian,
      Indian, and Others (like Hispanic, Latino, Middle Eastern).
  [date&time] is in the format of yyyymmddHHMMSSFFF, showing the
      date and time an image was collected to UTKFace



  """
  _, filename = os.path.split(path)
  return int(filename.split("_")[-2])


def get_image_paths(path):
  """
  This function retrieves image paths and their corresponding labels

  """
  image_paths = []

  for fold in os.listdir(path):
    for image_file in os.listdir(os.path.join(path, fold)):
      if image_file.lower().endswith(".jpg"):
        path = os.path.join(path, fold, image_file)
        image_paths.append(path)

  return image_paths


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True,
                        help="Path to UTK-Face dataset. \nIt should contain folders `part1`, `part2` and `part3`")
    parser.add_argument("-o", "--output_path", required=True, help="Path to save split dataset!")
    args = parser.parse_args()
    return args


def save_split(save_folder, image_paths, index2class):
    for i in range(len(image_paths)):
        image_path = image_paths[i]
        
        image_label = get_race(image_path)
        _, filename = os.path.split(image_path) # split image path into parent foldr and it's filename
        output_folder = os.path.join(save_folder, index2class[image_label])
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        output_path = os.path.join(output_folder, filename)
        shutil.copy(image_path, output_path)
def main(args):

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    image_paths = get_image_paths(args.path)
    
    trainset, test_valid_set = train_test_split(image_paths, test_size = 0.4)
    
    testset, valid_set = train_test_split(test_valid_set, test_size = 0.5)
    
    index2class = ["White", "Black", "Asian", "Indian", "Others"]
    class2index = {index2class[i]:i for i in range(len(index2class))}

    train_path = os.path.join(args.output, "train")
    test_path = os.path.join(args.output, "test")
    valid_path = os.path.join(args.output, "valid")
    
    save_split(train_path, trainset, index2class)
    save_split(test_path, testset, index2class)
    save_split(valid_path, validset, index2class)
    
    
    
    


if __name__ == '__main__':
    args = get_args()
    main(args)
    