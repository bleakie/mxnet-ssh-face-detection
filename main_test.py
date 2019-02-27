# -*- coding: utf-8 -*
import os
import cv2
import numpy as np
from ssh_detector import SSHDetector
import matplotlib
matplotlib.use('Agg')
from rcnn.config import config


detector = SSHDetector('./model/e2e', 0)

def get_all_path(IMAGE_DIR):
    image_path = []
    g = os.walk(IMAGE_DIR)
    for path, d, filelist in g:
        for filename in filelist:
            list = []
            if filename.endswith('jpg'):
                list.append(({"name": filename}, {"path": os.path.join(path, filename)}))
                image_path.append(list)
    return image_path


def get_boxes(im, pyramid, rate, thresh):
  # prevent bigger axis from being more than max_size:
  if not pyramid:
    scales = [1.0]#使用原始图片
  elif rate:
    scales = config.TEST.PYRAMID_SCALES #切换到pyramidbox的scale方法
  else:
    target_size = 720
    max_size = 1280
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    scales = [float(scale)/target_size*im_scale for scale in config.TEST.SCALES]
  print ('scale is: ', scales)
  boxes = detector.detect(im, threshold = thresh, scales = scales)#
  return boxes


def save_boxes(im, bboxes):
    if bboxes.shape[0] != 0:
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, :]
            p1 = (bbox[0], bbox[1])
            p2 = (bbox[2], bbox[3])
            cv2.rectangle(im, p1, p2, (0, 255, 0), 1)
            p3 = (max(p1[0], 15), max(p1[1], 15))
            title = "%.2f" % (bbox[4])
            cv2.putText(im, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
    return im

file_names = get_all_path('./image/test')
for index in range(len(file_names)):
    name = file_names[index][0][0]["name"]
    file_path = file_names[index][0][1]["path"]
    img = cv2.imread(file_path)

    faces = get_boxes(img, True, True, config.TEST.SCORE_THRESH)
    img = save_boxes(img, faces)
    cv2.imshow("cv", img)
    cv2.waitKey()


