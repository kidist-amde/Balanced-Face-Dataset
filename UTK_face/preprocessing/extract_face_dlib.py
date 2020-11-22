# -*- coding: utf-8 -*-
"""
This script crop face images.
"""

import dlib

detector = dlib.get_frontal_face_detector()

import os
import numpy as np
import pandas as pd
import PIL
from PIL import Image
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True,
                        help="Path to split UTK-Face dataset. \nIt should contain folders `train`, `test` and `valid`")
    parser.add_argument("-o", "--output_path", required=True, help="Path to save cropped face dataset!")
    args = parser.parse_args()
    return args


def get_largest_face(face_rectangles):

  if len(face_rectangles)==0:
    return None
  elif len(face_rectangles)==1:
    return face_rectangles[0]
  face_rec = face_rectangles[0]
  fw = face_rec.right() - face_rec.left()
  fh = face_rec.bottom() - face_rec.top()
  max_area = fw*fh
  max_index = 0
  for i in range(len(face_rectangles)):
    face_rec = face_rectangles[i]
    fw = face_rec.right() - face_rec.left()
    fh = face_rec.bottom() - face_rec.top()
    area = fw*fh
    if area >max_area:
      max_area = area
      max_index = i

  return face_rectangles[max_index]

def crop_face(detector, image_path):
  """
  Crop face image from image file containing face.

  Parameters
  ----------
  detector : dlib.fhog_object_detector
    Dlib frontal face detector
  image_path : str  
    Path to image file

  Returns
  -------
  numpy.ndarray
    Face image if dlib ables to find a face or the original image otherwise

  """
  image = Image.open(image_path)
  image = np.array(image)
  try:
    face_rectangles = detector(image)
  except:
    print("Error: when detecting face from: {}".format(image_path))
    # return image
    return None

  if len(face_rectangles) == 0:
    print("Warnining: Dlib was unable to detect face from: {}".format(image_path))
    # return image
    return None
  face_rectangle = face_rectangles[0]
  
  fw = face_rectangle.right() - face_rectangle.left()
  fh = face_rectangle.bottom() - face_rectangle.top()
  
  add_size = max(fw, fh)

  left = max(face_rectangle.left()-add_size, 0)
  top = max(face_rectangle.top()-add_size, 0)
  right = min(face_rectangle.right()+add_size, image.shape[1] )
  bottom = min(face_rectangle.bottom()+add_size, image.shape[0] )

  
  face_image = image[top:bottom, left:right]
  
  if np.prod(face_image.shape) == 0:
    return image
  return face_image


def crop_faces_folder(detector, source_folder, dest_folder, size):
  """
  Crops face images for all image files inside source folder and saves them to destination folder.
  The images are also resized to fixed size
  Parameters
  ----------
  source_folder: str
    A foder containing image files with face
  dest_folder: str
    Destination folder to save extracted face images
  size: tuple
    Size of face image 
  """
  for image_file in os.listdir(source_folder):
    image_path = os.path.join(source_folder, image_file)
    face_array = crop_face(detector, image_path)
    if face_array is None:
      continue
    face_image = Image.fromarray(face_array)
    face_image = face_image.resize(size)

    if face_array.shape[-1]==4:
      face_image = face_image.convert('RGB')


    output_path = os.path.join(dest_folder, image_file)
    face_image.save(output_path)


def crop_and_save_faces(detector, source_base_folder, dest_base_folder, size):
  for case in ["train", "valid", "test"]:
    print("Processing {} files".format(case))
    for class_ in os.listdir(os.path.join(source_base_folder, case)):
      source_folder = os.path.join(source_base_folder, case, class_)
      dest_folder = os.path.join(dest_base_folder, case, class_)
      if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
      crop_faces_folder(detector, source_folder, dest_folder, size)


def pad_face(image_array, face_rec):
  fw = face_rec.right() - face_rec.left()
  fh = face_rec.bottom() - face_rec.top()
  
  r = max(fw, fh)

  left = max(face_rec.left()-r, 0)
  top = max(face_rec.top()-r, 0)
  right = min(face_rec.right()+r, image_array.shape[1] )
  bottom = min(face_rec.bottom()+r, image_array.shape[0] )

  return image_array[top:bottom, left:right]
def main(args):
    if not os.path.exists(args.output_path):
      os.mkdir(args.output_path)
    size = (224, 224)
    crop_and_save_faces(detector, args.path, args.output_path, size)

if __name__ == '__main__':
    args = get_args()
    main(args)
    

