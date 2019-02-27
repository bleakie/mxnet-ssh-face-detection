from __future__ import print_function
import sys
import cv2
import mxnet as mx
from mxnet import ndarray as nd
import numpy as np
import numpy.random as npr
from distutils.util import strtobool
from rcnn.config import config

from rcnn.processing.bbox_transform import nonlinear_pred, clip_boxes
from rcnn.processing.generate_anchor import generate_anchors_fpn, anchors_plane
from rcnn.processing.nms import gpu_nms_wrapper


class SSHDetector:
  def __init__(self, prefix, epoch, ctx_id=0, test_mode=False):
    self.ctx_id = ctx_id
    self.ctx = mx.gpu(self.ctx_id)
    self.fpn_keys = []
    fpn_stride = []
    fpn_base_size = []
    self._feat_stride_fpn = [32, 16, 8]

    for s in self._feat_stride_fpn:
        self.fpn_keys.append('stride%s'%s)
        fpn_stride.append(int(s))
        fpn_base_size.append(16)

    self._scales = np.array([32,16,8,4,2,1])
    self._ratios = np.array([1.0]*len(self._feat_stride_fpn))
    self._anchors_fpn = dict(zip(self.fpn_keys, generate_anchors_fpn(base_size=fpn_base_size, scales=self._scales, ratios=self._ratios)))
    self._num_anchors = dict(zip(self.fpn_keys, [anchors.shape[0] for anchors in self._anchors_fpn.values()]))
    self._rpn_pre_nms_top_n = 1000
    #self._rpn_post_nms_top_n = rpn_post_nms_top_n
    #self.score_threshold = 0.05
    self.nms_threshold = config.TEST.NMS
    self._bbox_pred = nonlinear_pred
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    self.nms = gpu_nms_wrapper(self.nms_threshold, self.ctx_id)
    self.pixel_means = np.array([103.939, 116.779, 123.68]) #BGR

    if not test_mode:
      image_size = (640, 640)
      self.model = mx.mod.Module(symbol=sym, context=self.ctx, label_names = None)
      self.model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))], for_training=False)
      self.model.set_params(arg_params, aux_params)
    else:
      from rcnn.core.module import MutableModule
      image_size = (1280, 720)
      data_shape = [('data', (1,3,image_size[0], image_size[1]))]
      self.model = MutableModule(symbol=sym, data_names=['data'], label_names=None,
                                context=self.ctx, max_data_shapes=data_shape)
      self.model.bind(data_shape, None, for_training=False)
      self.model.set_params(arg_params, aux_params)


  def detect(self, img, threshold=0.5, scales=[1.0]):
    proposals_list = []
    scores_list = []

    im_src = img.copy()

    CONSTANT = config.TEST.CONSTANT
    BLACK = [0, 0, 0]
    img = cv2.copyMakeBorder(img, CONSTANT, CONSTANT, CONSTANT, CONSTANT, cv2.BORDER_CONSTANT, value=BLACK)
    #add by sai
    max_im_shrink = (0x7fffffff / 200.0 / (
            img.shape[0] * img.shape[1])) ** 0.5  # the max size of input image for caffe
    max_im_shrink = 3 if max_im_shrink > 3 else max_im_shrink

    for im_scale in scales:
      if (im_scale > max_im_shrink):#add by sai with filt big img
          continue
      if im_scale!=1.0:
        im = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
      else:
        im = img
      im = im.astype(np.float32)
      #self.model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))], for_training=False)
      im_info = [im.shape[0], im.shape[1], im_scale]
      im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]))
      for i in range(3):
          im_tensor[0, i, :, :] = im[:, :, 2 - i] - self.pixel_means[2 - i]
      data = nd.array(im_tensor)
      db = mx.io.DataBatch(data=(data,), provide_data=[('data', data.shape)])
      self.model.forward(db, is_train=False)
      net_out = self.model.get_outputs()
      pre_nms_topN = self._rpn_pre_nms_top_n

      for s in self._feat_stride_fpn:
          if len(scales)>1 and s==32 and im_scale==scales[-1]:
            continue
          _key = 'stride%s'%s
          stride = int(s)
          idx = 0
          if s==16:
            idx=2
          elif s==8:
            idx=4
          scores = net_out[idx].asnumpy()
          #print(scores.shape)
          idx+=1
          scores = scores[:, self._num_anchors['stride%s'%s]:, :, :]
          bbox_deltas = net_out[idx].asnumpy()

          _height, _width = int(im_info[0] / stride), int(im_info[1] / stride)
          height, width = bbox_deltas.shape[2], bbox_deltas.shape[3]

          A = self._num_anchors['stride%s'%s]
          K = height * width

          anchors = anchors_plane(height, width, stride, self._anchors_fpn['stride%s'%s].astype(np.float32))
          anchors = anchors.reshape((K * A, 4))

          bbox_deltas = self._clip_pad(bbox_deltas, (height, width))
          bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))

          scores = self._clip_pad(scores, (height, width))
          scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

          #print(anchors.shape, bbox_deltas.shape, A, K, file=sys.stderr)
          proposals = self._bbox_pred(anchors, bbox_deltas)
          #proposals = anchors

          proposals = clip_boxes(proposals, im_info[:2])

          scores_ravel = scores.ravel()
          order = scores_ravel.argsort()[::-1]
          if pre_nms_topN > 0:
              order = order[:pre_nms_topN]
          proposals = proposals[order, :]
          scores = scores[order]

          proposals /= im_scale

          #add by sai with pyramidbox to filt scale face
          if im_scale > 1:
              index = np.where(
                  np.minimum(proposals[:, 2] - proposals[:, 0] + 1,
                             proposals[:, 3] - proposals[:, 1] + 1) < 50)[0]
              proposals = proposals[index, :]
              scores = scores[index, :]
          else:
              index = np.where(
                  np.maximum(proposals[:, 2] - proposals[:, 0] + 1,
                             proposals[:, 3] - proposals[:, 1] + 1) > 20)[0]
              proposals = proposals[index, :]
              scores = scores[index, :]

          proposals_list.append(proposals)
          scores_list.append(scores)
    proposals = np.vstack(proposals_list)
    scores = np.vstack(scores_list)
    scores_ravel = scores.ravel()
    order = scores_ravel.argsort()[::-1]

    proposals = proposals[order, :]
    scores = scores[order]

    det = np.hstack((proposals, scores)).astype(np.float32)

    if self.nms_threshold<1.0:
      keep = self.nms(det)
      det = det[keep, :]
    if threshold>0.0:
      keep = np.where(det[:, 4] >= threshold)[0]
      det = det[keep, :]

      # add by sai
    if det.shape[0] != 0:
        for i in range(det.shape[0]):
            det[i, :][0] = det[i, :][0] - CONSTANT
            det[i, :][1] = det[i, :][1] - CONSTANT
            det[i, :][2] = det[i, :][2] - CONSTANT
            det[i, :][3] = det[i, :][3] - CONSTANT
            if det[i, :][0] < 0:
                det[i, :][0] = 0
            if det[i, :][2] > im_src.shape[1]:
                det[i, :][2] = im_src.shape[1]
            if det[i, :][1] < 0:
                det[i, :][1] = 0
            if det[i, :][3] > im_src.shape[0]:
                det[i, :][3] = im_src.shape[0]
    return det

  @staticmethod
  def _filter_boxes(boxes, min_size):
      """ Remove all boxes with any side smaller than min_size """
      ws = boxes[:, 2] - boxes[:, 0] + 1
      hs = boxes[:, 3] - boxes[:, 1] + 1
      keep = np.where((ws >= min_size) & (hs >= min_size))[0]
      return keep

  @staticmethod
  def _clip_pad(tensor, pad_shape):
      """
      Clip boxes of the pad area.
      :param tensor: [n, c, H, W]
      :param pad_shape: [h, w]
      :return: [n, c, h, w]
      """
      H, W = tensor.shape[2:]
      h, w = pad_shape

      if h < H or w < W:
        tensor = tensor[:, :, :h, :w].copy()

      return tensor