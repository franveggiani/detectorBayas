from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
import torch
import _init_paths

import os
import cv2

from opts import opts
from detectors.detector_factory import detector_factory
import json

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']


def get_directories(path):
    """
    Retorna una lista con los nombres de los directorios en la ruta especificada.
    """
    # Obtenemos una lista con todos los elementos en la ruta
    elementos = os.listdir(path)

    # Filtramos los elementos que sean directorios
    directorios = [elem for elem in elementos if os.path.isdir(os.path.join(path, elem))]

    return directorios

def create_output_directories(root, directories_list):
  path_of, of = os.path.split(root[:-1])
  path_to_look, _ = os.path.splitext(path_of)
  for directory in directories_list:

    if not os.path.isdir(root+directory) and os.path.exists(path_to_look+'/'+directory+'/input.txt'):
      os.mkdir(root+directory)


def process_videos(opt):

  cam = cv2.VideoCapture(opt.demo)
  opt.detector_for_track.pause = False
  video_path, video_name_with_extension = os.path.split(opt.demo)
  video_name, extension = os.path.splitext(video_name_with_extension)
  _ , output_folder_name = os.path.split(video_path)
  output_folder_name = output_folder_name + '/'
  frame_number=0
  flag = True
  dets = dict()
  while flag:
    flag, img = cam.read()
    if not flag :#or fr == 300
      break
    # cv2.imshow('input', img)
    ret = opt.detector_for_track.run_det_for_byte(img)[1]
    ret = ret.astype(np.float)
    ret = filter(lambda x: x[0]>0 and x[1]>0, ret)
    dets[frame_number] = { k:list(r[:3]) for k,r in enumerate(ret)}
    frame_number += 1
    print(f'video:{video_name}, frame number: {frame_number}')

  json.dump(dets, open(opt.output_folder_json + output_folder_name + video_name + '.json', 'w'))




if __name__ == '__main__':
  torch.backends.cudnn.enabled = False
  opt = opts().init()
  input_path = opt.input_folder
  directories = get_directories(input_path)
  #
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.debug = max(opt.debug, 1)
  Detector = detector_factory[opt.task]
  opt.detector_for_track = Detector(opt)
  create_output_directories(opt.output_folder_json, directories)
  for directory in directories:
    try:
      input_file = open(input_path+directory+'/input.txt', 'r')
    except:
      print(f"El directorio {directory} no contiene archivo input.txt")
      continue
    for line in input_file:
      opt.demo = input_path + directory + '/' + line[:-1]
      process_videos(opt)


