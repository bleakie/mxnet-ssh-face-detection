from __future__ import print_function
import numpy as np
import cv2
import os
import math
import sys
import random
from ..config import config
#add by sai whit hsv from smallhardface
import numpy.random as npr

def brightness_aug(src, x):
  alpha = 1.0 + random.uniform(-x, x)
  src *= alpha
  return src

def contrast_aug(src, x):
  alpha = 1.0 + random.uniform(-x, x)
  coef = np.array([[[0.299, 0.587, 0.114]]])
  gray = src * coef
  gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
  src *= alpha
  src += gray
  return src

def saturation_aug(src, x):
  alpha = 1.0 + random.uniform(-x, x)
  coef = np.array([[[0.299, 0.587, 0.114]]])
  gray = src * coef
  gray = np.sum(gray, axis=2, keepdims=True)
  gray *= (1.0 - alpha)
  src *= alpha
  src += gray
  return src
# add augmentation in random, and only once time
def color_aug(img, x):
  augs = [brightness_aug, contrast_aug, saturation_aug]
  random.shuffle(augs)
  for aug in augs:
    #print(img.shape)
    img = aug(img, x)
    #print(img.shape)
  return img

#add blur
def blur_aug(img):
    augs = ['GaussianBlur', 'MedianBlur', 'AverageBlur']
    random.shuffle(augs)
    index = augs[0]
    if index == 'GaussianBlur':
        img = cv2.GaussianBlur(img, (3, 3), 0)
    elif index == 'MedianBlur':
        img = cv2.medianBlur(img, 5)
    else:
        img = cv2.blur(img, (3, 3))
    return img

#add augmentation with hsv from smallhardface
def hue_aug(hsv_im, x):  # inplace transformation
    if npr.rand() < 0.5:
        delta = npr.uniform(-x,  x)
        hsv_im[..., 0] = (hsv_im[..., 0] + delta) % 360.0
    return hsv_im

# add augmentation in random as smallhardface, and sometimes
def distortion(im, x):
    im = brightness_aug(im, x)
    im = blur_aug(im)
    if npr.randint(2):
        im = contrast_aug(im, x)
        hsv_im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        hsv_im = saturation_aug(hsv_im, x)
        hsv_im = hue_aug(hsv_im, x)
        im = cv2.cvtColor(hsv_im, cv2.COLOR_HSV2BGR)
    else:
        hsv_im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        hsv_im = saturation_aug(hsv_im, x)
        hsv_im = hue_aug(hsv_im, x)
        im = cv2.cvtColor(hsv_im, cv2.COLOR_HSV2BGR)
        im = contrast_aug(im, x)
    im = np.clip(np.round(im), 0.0, 255.0)
    return im

def get_image(roidb, scale=False):
    """
    preprocess image and return processed roidb
    :param roidb: a list of roidb
    :return: list of img as in mxnet format
    roidb add new item['im_info']
    0 --- x (width, second dim of im)
    |
    y (height, first dim of im)
    """
    num_images = len(roidb)
    processed_ims = []
    processed_roidb = []
    for i in range(num_images):
        roi_rec = roidb[i]
        if 'stream' in roi_rec:
          im = cv2.imdecode(roi_rec['stream'], cv2.IMREAD_COLOR)
        else:
          assert os.path.exists(roi_rec['image']), '{} does not exist'.format(roi_rec['image'])
          im = cv2.imread(roi_rec['image'])
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        new_rec = roi_rec.copy()
        if scale:
          scale_range = config.TRAIN.SCALE_RANGE
          im_scale = np.random.uniform(scale_range[0], scale_range[1])
          im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        elif not config.ORIGIN_SCALE:
          scale_ind = random.randrange(len(config.SCALES))
          target_size = config.SCALES[scale_ind][0]
          max_size = config.SCALES[scale_ind][1]
          im, im_scale = resize(im, target_size, max_size, stride=config.IMAGE_STRIDE)
        else:
          im_scale = 1.0
        im_tensor = transform(im, config.PIXEL_MEANS)
        if 'boxes_mask' in roi_rec:
          im = im.astype(np.float32)
          boxes_mask = roi_rec['boxes_mask'].copy() * im_scale
          boxes_mask = boxes_mask.astype(np.int)
          for j in xrange(boxes_mask.shape[0]):
            m = boxes_mask[j]
            im_tensor[:,:,m[1]:m[3],m[0]:m[2]] = 0.0
            #print('find mask', m, file=sys.stderr)
        processed_ims.append(im_tensor)
        new_rec['boxes'] = roi_rec['boxes'].copy() * im_scale
        if config.TRAIN.IMAGE_ALIGN>0:
          if im_tensor.shape[2]%config.TRAIN.IMAGE_ALIGN!=0 or im_tensor.shape[3]%config.TRAIN.IMAGE_ALIGN!=0:
            new_height = math.ceil(float(im_tensor.shape[2])/config.TRAIN.IMAGE_ALIGN)*config.TRAIN.IMAGE_ALIGN
            new_width = math.ceil(float(im_tensor.shape[3])/config.TRAIN.IMAGE_ALIGN)*config.TRAIN.IMAGE_ALIGN
            new_im_tensor = np.zeros((1, 3, int(new_height), int(new_width)))
            new_im_tensor[:,:,0:im_tensor.shape[2],0:im_tensor.shape[3]] = im_tensor
            print(im_tensor.shape, new_im_tensor.shape, file=sys.stderr)
            im_tensor = new_im_tensor
        #print('boxes', new_rec['boxes'], file=sys.stderr)
        im_info = [im_tensor.shape[2], im_tensor.shape[3], im_scale]
        new_rec['im_info'] = im_info
        processed_roidb.append(new_rec)
    return processed_ims, processed_roidb

TMP_ID = 0

def get_crop_image(roidb):
    """
    preprocess image and return processed roidb
    :param roidb: a list of roidb
    :return: list of img as in mxnet format
    roidb add new item['im_info']
    0 --- x (width, second dim of im)
    |
    y (height, first dim of im)
    """
    #roidb and each roi_rec can not be changed as it will be reused in next epoch
    num_images = len(roidb)
    processed_ims = []
    processed_roidb = []
    for i in range(num_images):
        roi_rec = roidb[i]
        if 'stream' in roi_rec:
          im = cv2.imdecode(roi_rec['stream'], cv2.IMREAD_COLOR)
        else:
          assert os.path.exists(roi_rec['image']), '{} does not exist'.format(roi_rec['image'])
          im = cv2.imread(roi_rec['image'])
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        if 'boxes_mask' in roi_rec:
          #im = im.astype(np.float32)
          boxes_mask = roi_rec['boxes_mask'].copy()
          boxes_mask = boxes_mask.astype(np.int)
          for j in xrange(boxes_mask.shape[0]):
            m = boxes_mask[j]
            im[m[1]:m[3],m[0]:m[2],:] = (104, 117, 123)
        scale_ind = random.randrange(len(config.SCALES))
        SIZE = config.SCALES[scale_ind][0]# config.SCALES[scale_ind][0]
        PRE_SCALES = [0.3, 0.45, 0.6, 0.8, 1.0]
        _scale = random.choice(PRE_SCALES)
        #_scale = np.random.uniform(PRE_SCALES[0], PRE_SCALES[-1])
        size = int(np.round(_scale*np.min(im.shape[0:2])))
        im_scale = float(SIZE)/size
        #origin_im_scale = im_scale
        #size = np.round(np.min(im.shape[0:2])*im_scale)
        #im_scale *= (float(SIZE)/size)
        origin_shape = im.shape
        if _scale>10.0: #avoid im.size<SIZE, never?
          sizex = int(np.round(im.shape[1]*im_scale))
          sizey = int(np.round(im.shape[0]*im_scale))
          if sizex<SIZE:
            sizex = SIZE
            print('keepx', sizex)
          if sizey<SIZE:
            sizey = SIZE
            print('keepy', sizex)
          im = cv2.resize(im, (sizex, sizey), interpolation=cv2.INTER_LINEAR)
        else:
          im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        assert im.shape[0]>=SIZE and im.shape[1]>=SIZE

        new_rec = roi_rec.copy()
        new_rec['boxes'] = roi_rec['boxes'].copy() * im_scale
        retry = 0
        LIMIT = 25
        size = SIZE
        while retry<LIMIT:
          up, left = (np.random.randint(0, im.shape[0]-size+1), np.random.randint(0, im.shape[1]-size+1))
          boxes_new = new_rec['boxes'].copy()
          im_new = im[up:(up+size), left:(left+size), :]
          #print('crop', up, left, size, im_scale)
          boxes_new[:,0] -= left
          boxes_new[:,2] -= left
          boxes_new[:,1] -= up
          boxes_new[:,3] -= up
          #im_new = cv2.resize(im_new, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
          #boxes_new *= im_scale
          #print(origin_shape, im_new.shape, im_scale)
          valid = []
          valid_boxes = []
          for i in xrange(boxes_new.shape[0]):
            box = boxes_new[i]
            center = np.array(([box[0], box[1]]+[box[2], box[3]]))/2

            #box[0] = max(0, box[0])
            #box[1] = max(0, box[1])
            #box[2] = min(im_new.shape[1], box[2])
            #box[3] = min(im_new.shape[0], box[3])
            box_size = max(box[2]-box[0], box[3]-box[1])

            if center[0]<0 or center[1]<0 or center[0]>=im_new.shape[1] or center[1]>=im_new.shape[0]:
              continue
            if box_size<config.TRAIN.MIN_BOX_SIZE:
              continue
            valid.append(i)
            valid_boxes.append(box)
          if len(valid)>0 or retry==LIMIT-1:
            im = im_new
            new_rec['boxes'] = np.array(valid_boxes)
            new_rec['gt_classes'] = new_rec['gt_classes'][valid]
            break

          retry+=1

        if config.COLOR_JITTERING>0.0:
          im = im.astype(np.float32)
          im = color_aug(im, config.COLOR_JITTERING)#in original aug
          # im = distortion(im, config.COLOR_JITTERING)  # in smallhardface aug

        DEBUG = False
        if DEBUG:
          global TMP_ID
          if TMP_ID<10:
            tim = im.copy().astype(np.uint8)
            for i in xrange(new_rec['boxes'].shape[0]):
              box = new_rec['boxes'][i].copy().astype(np.int)
              cv2.rectangle(tim, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 1)
            filename = './tmp/D%d.png' % TMP_ID
            print('write', filename)
            TMP_ID+=1
            cv2.imwrite(filename, tim)

        im_tensor = transform(im, config.PIXEL_MEANS)

        processed_ims.append(im_tensor)
        #print('boxes', new_rec['boxes'], file=sys.stderr)
        im_info = [im_tensor.shape[2], im_tensor.shape[3], im_scale]
        new_rec['im_info'] = im_info
        processed_roidb.append(new_rec)
    return processed_ims, processed_roidb

def resize(im, target_size, max_size, stride=0, min_size=0):
    """
    only resize input image to target size and return scale
    :param im: BGR image input by opencv
    :param target_size: one dimensional size (the short side)
    :param max_size: one dimensional max size (the long side)
    :param stride: if given, pad the image to designated stride
    :return:
    """
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
        if min_size>0 and np.round(im_scale*im_size_min)<min_size:
          im_scale = float(min_size) / float(im_size_min)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

    if stride == 0:
        return im, im_scale
    else:
        # pad to product of stride
        im_height = int(np.ceil(im.shape[0] / float(stride)) * stride)
        im_width = int(np.ceil(im.shape[1] / float(stride)) * stride)
        im_channel = im.shape[2]
        padded_im = np.zeros((im_height, im_width, im_channel))
        padded_im[:im.shape[0], :im.shape[1], :] = im
        return padded_im, im_scale


def transform(im, pixel_means):
    """
    transform into mxnet tensor,
    subtract pixel size and transform to correct format
    :param im: [height, width, channel] in BGR
    :param pixel_means: [B, G, R pixel means]
    :return: [batch, channel, height, width]
    """
    im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]))
    for i in range(3):
        im_tensor[0, i, :, :] = im[:, :, 2 - i] - pixel_means[2 - i]
    return im_tensor


def transform_inverse(im_tensor, pixel_means):
    """
    transform from mxnet im_tensor to ordinary RGB image
    im_tensor is limited to one image
    :param im_tensor: [batch, channel, height, width]
    :param pixel_means: [B, G, R pixel means]
    :return: im [height, width, channel(RGB)]
    """
    assert im_tensor.shape[0] == 1
    im_tensor = im_tensor.copy()
    # put channel back
    channel_swap = (0, 2, 3, 1)
    im_tensor = im_tensor.transpose(channel_swap)
    im = im_tensor[0]
    assert im.shape[2] == 3
    im += pixel_means[[2, 1, 0]]
    im = im.astype(np.uint8)
    return im


def tensor_vstack(tensor_list, pad=0):
    """
    vertically stack tensors
    :param tensor_list: list of tensor to be stacked vertically
    :param pad: label to pad with
    :return: tensor with max shape
    """
    ndim = len(tensor_list[0].shape)
    dtype = tensor_list[0].dtype
    islice = tensor_list[0].shape[0]
    dimensions = []
    first_dim = sum([tensor.shape[0] for tensor in tensor_list])
    dimensions.append(first_dim)
    for dim in range(1, ndim):
        dimensions.append(max([tensor.shape[dim] for tensor in tensor_list]))
    if pad == 0:
        all_tensor = np.zeros(tuple(dimensions), dtype=dtype)
    elif pad == 1:
        all_tensor = np.ones(tuple(dimensions), dtype=dtype)
    else:
        all_tensor = np.full(tuple(dimensions), pad, dtype=dtype)
    if ndim == 1:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice] = tensor
    elif ndim == 2:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice, :tensor.shape[1]] = tensor
    elif ndim == 3:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice, :tensor.shape[1], :tensor.shape[2]] = tensor
    elif ndim == 4:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice, :tensor.shape[1], :tensor.shape[2], :tensor.shape[3]] = tensor
    elif ndim == 5:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice, :tensor.shape[1], :tensor.shape[2], :tensor.shape[3], :tensor.shape[4]] = tensor
    else:
        print(tensor_list[0].shape)
        raise Exception('Sorry, unimplemented.')
    return all_tensor

