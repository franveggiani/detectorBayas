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

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.debug = max(opt.debug, 1)
  Detector = detector_factory[opt.task]
  detector = Detector(opt)

  if opt.demo == 'webcam' or \
    opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
    cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
    detector.pause = False
    fr=0
    df_list = []
    flag = True
    while flag:
        flag, img = cam.read()
        if not flag :#or fr == 300
            break
        # cv2.imshow('input', img)
        ret = detector.run_det_for_byte(img)[1]
        dets = np.zeros((opt.K, 6))
        dets[:, 0] = ret[:, 0] - ret[:, 2]
        dets[:, 1] = ret[:, 1] - ret[:, 2]
        dets[:, 2] = ret[:, 0] + ret[:, 2]
        dets[:, 3] = ret[:, 1] + ret[:, 2]
        dets[:, 4] = ret[:, 3]
        dets[:, 5] = fr
        df = pd.DataFrame(dets, columns=["x1", "y1", "x2", "y2", "score", "frame"])
        df_list.append(df)
        fr += 1
    df = pd.concat(df_list)
    print(opt.demo[:-4])
    df.to_csv('../data/video/'+ opt.demo[14:-4]+'.csv')
  else:
    if os.path.isdir(opt.demo):
      image_names = []
      ls = os.listdir(opt.demo)
      for file_name in sorted(ls):
          ext = file_name[file_name.rfind('.') + 1:].lower()
          if ext in image_ext:
              image_names.append(os.path.join(opt.demo, file_name))
    else:
      image_names = [opt.demo]

    print(image_names)
    
    for image_name in image_names:
      ret = detector.run(image_name)
      time_str = ''
      for stat in time_stats:
        time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
      print(time_str)
if __name__ == '__main__':
  torch.backends.cudnn.enabled = False
  opt = opts().init()
  demo(opt)
