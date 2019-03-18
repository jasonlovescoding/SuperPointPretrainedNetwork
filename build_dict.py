import argparse
import glob
import numpy as np
import os
import time

import cv2
import torch
import sklearn
import scipy

from model import SuperPointFrontend
from util import read_image 

if __name__ == '__main__':
  # Parse command line arguments.
  parser = argparse.ArgumentParser(description='PyTorch SuperPoint-Based Image Comparison.')
  parser.add_argument('data_path', type=str,
      help='Path to the directory of the dataset')
  parser.add_argument('--save_path', type=str, default='superpoint_v1.npy',
      help='Path to save the dictionary')
  parser.add_argument('--weights_path', type=str, default='superpoint_v1.pth',
      help='Path to pretrained weights file (default: superpoint_v1.pth).')
  parser.add_argument('--H', type=int, default=120,
      help='Input image height (default: 120).')
  parser.add_argument('--W', type=int, default=160,
      help='Input image width (default:160).')
  parser.add_argument('--batch_size', type=int, default=128,
      help='Input batch size (default: 128).')
  parser.add_argument('--num_workers', type=int, default=4,
      help='Dataloader number of workers (default: 4)')
  parser.add_argument('--nms_dist', type=int, default=4,
      help='Non Maximum Suppression (NMS) distance (default: 4).')
  parser.add_argument('--conf_thresh', type=float, default=0.015,
      help='Detector confidence threshold (default: 0.015).')
  parser.add_argument('--nn_thresh', type=float, default=0.7,
      help='Descriptor matching threshold (default: 0.7).')
  parser.add_argument('--cuda', action='store_true',
      help='Use cuda GPU to speed up network processing speed (default: False)')
  opt = parser.parse_args()
  print(opt)

  print('==> Loading pre-trained network.')
  # This class runs the SuperPoint network and processes its outputs.
  fe = SuperPointFrontend(weights_path=opt.weights_path,
                          nms_dist=opt.nms_dist,
                          conf_thresh=opt.conf_thresh,
                          nn_thresh=opt.nn_thresh,
                          cuda=opt.cuda)
  print('==> Successfully loaded pre-trained network.')

  print('==> Building visual dictionary.')
  bovw_dict = []
  i = 0
  for img in os.listdir(opt.data_path):
    data = read_image(opt.data_path + '/' + img, (opt.H, opt.W))
    pts, desc, heatmap = fe.run(data)
    bovw_dict.append(desc) 
    i += 1
    if i % 100 == 0:
        print('Image iteration reaches {} ...'.format(i))
  bovw_dict = np.concatenate(bovw_dict, axis=1)
  np.save(opt.save_path, bovw_dict)
  print('==> Successfully built visual dictionary.')
