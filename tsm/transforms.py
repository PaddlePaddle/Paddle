# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import traceback
import numpy as np
from PIL import Image

import logging
logger = logging.getLogger(__name__)

__all__ = ['GroupScale', 'GroupMultiScaleCrop', 'GroupRandomCrop',
           'GroupRandomFlip', 'GroupCenterCrop', 'NormalizeImage',
           'Compose']


class Compose(object):
    def __init__(self, transforms=[]):
        self.transforms = transforms

    def __call__(self, *data):
        for f in self.transforms:
            try:
                data = f(*data)
            except Exception as e:
                stack_info = traceback.format_exc()
                logger.info("fail to perform transform [{}] with error: "
                        "{} and stack:\n{}".format(f, e, str(stack_info)))
                raise e
        return data


class GroupScale(object):
    """
    Group scale image

    Args:
        target_size (int): image resize target size
    """
    def __init__(self, target_size=224):
        self.target_size = target_size

    def __call__(self, imgs, label):
	resized_imgs = []
	for i in range(len(imgs)):
	    img = imgs[i]
	    w, h = img.size
	    if (w <= h and w == self.target_size) or \
                    (h <= w and h == self.target_size):
		resized_imgs.append(img)
		continue

	    if w < h:
		ow = self.target_size
		oh = int(self.target_size * 4.0 / 3.0)
		resized_imgs.append(img.resize((ow, oh), Image.BILINEAR))
	    else:
		oh = self.target_size
		ow = int(self.target_size * 4.0 / 3.0)
		resized_imgs.append(img.resize((ow, oh), Image.BILINEAR))

	return resized_imgs, label


class GroupMultiScaleCrop(object):
    """
    FIXME: add comments
    """
    def __init__(self,
                 short_size=256,
                 scales=None,
                 max_distort=1,
                 fix_crop=True,
                 more_fix_crop=True):
        self.short_size = short_size
        self.scales = scales if scales is not None \
                        else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop

    def __call__(self, imgs, label):
	input_size = [self.short_size, self.short_size]

	im_size = imgs[0].size

	# get random crop offset
	def _sample_crop_size(im_size):
	    image_w, image_h = im_size[0], im_size[1]

	    base_size = min(image_w, image_h)
	    crop_sizes = [int(base_size * x) for x in self.scales]
	    crop_h = [
		input_size[1] if abs(x - input_size[1]) < 3 else x
		for x in crop_sizes
	    ]
	    crop_w = [
		input_size[0] if abs(x - input_size[0]) < 3 else x
		for x in crop_sizes
	    ]

	    pairs = []
	    for i, h in enumerate(crop_h):
		for j, w in enumerate(crop_w):
		    if abs(i - j) <= self.max_distort:
			pairs.append((w, h))
	    crop_pair = random.choice(pairs)
	    if not self.fix_crop:
		w_offset = np.random.randint(0, image_w - crop_pair[0])
		h_offset = np.random.randint(0, image_h - crop_pair[1])
	    else:
		w_step = (image_w - crop_pair[0]) / 4
		h_step = (image_h - crop_pair[1]) / 4

		ret = list()
		ret.append((0, 0))  # upper left
		if w_step != 0:
		    ret.append((4 * w_step, 0))  # upper right
		if h_step != 0:
		    ret.append((0, 4 * h_step))  # lower left
		if h_step != 0 and w_step != 0:
		    ret.append((4 * w_step, 4 * h_step))  # lower right
		if h_step != 0 or w_step != 0:
		    ret.append((2 * w_step, 2 * h_step))  # center

		if self.more_fix_crop:
		    ret.append((0, 2 * h_step))  # center left
		    ret.append((4 * w_step, 2 * h_step))  # center right
		    ret.append((2 * w_step, 4 * h_step))  # lower center
		    ret.append((2 * w_step, 0 * h_step))  # upper center

		    ret.append((1 * w_step, 1 * h_step))  # upper left quarter
		    ret.append((3 * w_step, 1 * h_step))  # upper right quarter
		    ret.append((1 * w_step, 3 * h_step))  # lower left quarter
		    ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

		w_offset, h_offset = random.choice(ret)

	    return crop_pair[0], crop_pair[1], w_offset, h_offset

	crop_w, crop_h, offset_w, offset_h = _sample_crop_size(im_size)
	crop_imgs = [
	    img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h))
	    for img in imgs
	]
	ret_imgs = [
	    img.resize((input_size[0], input_size[1]), Image.BILINEAR)
	    for img in crop_imgs
	]

	return ret_imgs, label


class GroupRandomCrop(object):
    def __init__(self, target_size=224):
        self.target_size = target_size

    def __call__(self, imgs, label):
	w, h = imgs[0].size
	th, tw = self.target_size, self.target_size

	assert (w >= self.target_size) and (h >= self.target_size), \
	      "image width({}) and height({}) should be larger than " \
              "crop size".format(w, h, self.target_size)

	out_images = []
	x1 = np.random.randint(0, w - tw)
	y1 = np.random.randint(0, h - th)

	for img in imgs:
	    if w == tw and h == th:
		out_images.append(img)
	    else:
		out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

	return out_images, label


class GroupRandomFlip(object):
    def __call__(self, imgs, label):
	v = np.random.random()
	if v < 0.5:
	    ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in imgs]
	    return ret, label
	else:
	    return imgs, label 


class GroupCenterCrop(object):
    def __init__(self, target_size=224):
        self.target_size = target_size

    def __call__(self, imgs, label):
	crop_imgs = []
	for img in imgs:
	    w, h = img.size
	    th, tw = self.target_size, self.target_size
	    assert (w >= self.target_size) and (h >= self.target_size), \
		 "image width({}) and height({}) should be larger " \
                 "than crop size".format(w, h, self.target_size)
	    x1 = int(round((w - tw) / 2.))
	    y1 = int(round((h - th) / 2.))
	    crop_imgs.append(img.crop((x1, y1, x1 + tw, y1 + th)))

	return crop_imgs, label 


class NormalizeImage(object):
    def __init__(self,
                 target_size=224,
                 img_mean=[0.485, 0.456, 0.406],
                 img_std=[0.229, 0.224, 0.225],
                 seg_num=8,
                 seg_len=1):
        self.target_size = target_size
        self.img_mean = np.array(img_mean).reshape((3, 1, 1)).astype('float32')
        self.img_std = np.array(img_std).reshape((3, 1, 1)).astype('float32')
        self.seg_num = seg_num
        self.seg_len = seg_len

    def __call__(self, imgs, label):
	np_imgs = (np.array(imgs[0]).astype('float32').transpose(
	    (2, 0, 1))).reshape(1, 3, self.target_size,
            self.target_size) / 255
	for i in range(len(imgs) - 1):
	    img = (np.array(imgs[i + 1]).astype('float32').transpose(
		(2, 0, 1))).reshape(1, 3, self.target_size,
                self.target_size) / 255
	    np_imgs = np.concatenate((np_imgs, img))

	np_imgs -= self.img_mean
	np_imgs /= self.img_std
	np_imgs = np.reshape(np_imgs, (self.seg_num, self.seg_len * 3,
                                 self.target_size, self.target_size))

        return np_imgs, label
