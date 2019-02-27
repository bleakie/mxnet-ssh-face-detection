# -*- coding: utf-8 -*
import os
import cv2
import numpy as np
from ssh_detector import SSHDetector
import matplotlib
matplotlib.use('Agg')
from rcnn.config import config

detector = SSHDetector('./model/e2e', 0)
image_dir = "./FDDB/originalPics/"
image_list = "./FDDB/output/faceList.txt"
image_dir_out = "./FDDB/output/"


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

# save img and boxes as json
def save_boxes(frame, bboxes, img_dir):
    count = 0
    _str = ""
    str_name = img_dir + "\n"
    str_box = ""
    _str += str_name
    if bboxes.shape[0] != 0:
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, :]
            p1 = (bbox[0], bbox[1])
            p2 = (bbox[2], bbox[3])
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            cv2.rectangle(frame, p1, p2, (0, 255, 0))
            p3 = (max(p1[0], 15), max(p1[1], 15))
            title = "%.2f" % (bbox[4])
            cv2.putText(frame, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
            str_box += str(bbox[0]) + " " \
                        + str(bbox[1]) + " " \
                        + str(bbox[2] - bbox[0]) + " " \
                        + str(bbox[3] - bbox[1]) + " " \
                        + str(bbox[4]) + "\n"
            count += 1
        _str += str(count) + "\n"
        _str += str_box
    else:
        _str += str(count) + "\n"
    return _str, frame

imgs_path_fd = open(image_list, "r")
imgs_path = imgs_path_fd.readlines()
imgs_path_fd.close()
str_ret = ""
for index in imgs_path:
    img_path = os.path.join(image_dir, index.strip("\r\n") + '.jpg')
    img = cv2.imread(img_path)

    faces = get_boxes(img, False, 0.67)
    _str, frame = save_boxes(img, faces, index.strip("\r\n"))
    str_ret += _str
    print(img_path)
    cv2.imshow("cv", img)
    cv2.waitKey(1)

d_ret_file = image_dir_out + "ssh-fddb-output.txt"
d_ret_fd = open(d_ret_file, "w")
d_ret_fd.writelines(str_ret)
d_ret_fd.close()
