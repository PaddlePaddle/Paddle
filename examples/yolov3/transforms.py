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

from __future__ import division
from __future__ import print_function

import cv2
import traceback
import numpy as np

__all__ = [
    "Compose",
    'ColorDistort',
    'RandomExpand',
    'RandomCrop',
    'RandomFlip',
    'NormalizeBox',
    'PadBox',
    'RandomShape',
    'NormalizeImage',
    'BboxXYXY2XYWH',
    'ResizeImage',
]


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *data):
        for f in self.transforms:
            try:
                data = f(*data)
            except Exception as e:
                stack_info = traceback.format_exc()
                print("fail to perform transform [{}] with error: "
                      "{} and stack:\n{}".format(f, e, str(stack_info)))
                raise e
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ColorDistort(object):
    """Random color distortion.

    Args:
        hue (list): hue settings.
            in [lower, upper, probability] format.
        saturation (list): saturation settings.
            in [lower, upper, probability] format.
        contrast (list): contrast settings.
            in [lower, upper, probability] format.
        brightness (list): brightness settings.
            in [lower, upper, probability] format.
        random_apply (bool): whether to apply in random (yolo) or fixed (SSD)
            order.
    """

    def __init__(self,
                 hue=[-18, 18, 0.5],
                 saturation=[0.5, 1.5, 0.5],
                 contrast=[0.5, 1.5, 0.5],
                 brightness=[0.5, 1.5, 0.5],
                 random_apply=True):
        self.hue = hue
        self.saturation = saturation
        self.contrast = contrast
        self.brightness = brightness
        self.random_apply = random_apply

    def apply_hue(self, img):
        low, high, prob = self.hue
        if np.random.uniform(0., 1.) < prob:
            return img

        img = img.astype(np.float32)

        # XXX works, but result differ from HSV version
        delta = np.random.uniform(low, high)
        u = np.cos(delta * np.pi)
        w = np.sin(delta * np.pi)
        bt = np.array([[1.0, 0.0, 0.0], [0.0, u, -w], [0.0, w, u]])
        tyiq = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.321],
                         [0.211, -0.523, 0.311]])
        ityiq = np.array([[1.0, 0.956, 0.621], [1.0, -0.272, -0.647],
                          [1.0, -1.107, 1.705]])
        t = np.dot(np.dot(ityiq, bt), tyiq).T
        img = np.dot(img, t)
        return img

    def apply_saturation(self, img):
        low, high, prob = self.saturation
        if np.random.uniform(0., 1.) < prob:
            return img
        delta = np.random.uniform(low, high)

        img = img.astype(np.float32)
        gray = img * np.array([[[0.299, 0.587, 0.114]]], dtype=np.float32)
        gray = gray.sum(axis=2, keepdims=True)
        gray *= (1.0 - delta)
        img *= delta
        img += gray
        return img

    def apply_contrast(self, img):
        low, high, prob = self.contrast
        if np.random.uniform(0., 1.) < prob:
            return img
        delta = np.random.uniform(low, high)

        img = img.astype(np.float32)
        img *= delta
        return img

    def apply_brightness(self, img):
        low, high, prob = self.brightness
        if np.random.uniform(0., 1.) < prob:
            return img
        delta = np.random.uniform(low, high)

        img = img.astype(np.float32)
        img += delta
        return img

    def __call__(self, im_id, im_shape, im, gt_bbox, gt_class, gt_score):
        if self.random_apply:
            distortions = np.random.permutation([
                self.apply_brightness, self.apply_contrast,
                self.apply_saturation, self.apply_hue
            ])
            for func in distortions:
                im = func(im)
            return [im_id, im_shape, im, gt_bbox, gt_class, gt_score]

        im = self.apply_brightness(im)

        if np.random.randint(0, 2):
            im = self.apply_contrast(im)
            im = self.apply_saturation(im)
            im = self.apply_hue(im)
        else:
            im = self.apply_saturation(im)
            im = self.apply_hue(im)
            im = self.apply_contrast(im)
        return [im_id, im_shape, im, gt_bbox, gt_class, gt_score]


class RandomExpand(object):
    """Random expand the canvas.

    Args:
        ratio (float): maximum expansion ratio.
        prob (float): probability to expand.
        fill_value (list): color value used to fill the canvas. in RGB order.
    """

    def __init__(self,
                 ratio=4.,
                 prob=0.5,
                 fill_value=[123.675, 116.28, 103.53]):
        assert ratio > 1.01, "expand ratio must be larger than 1.01"
        self.ratio = ratio
        self.prob = prob
        self.fill_value = fill_value

    def __call__(self, im_id, im_shape, im, gt_bbox, gt_class, gt_score):
        if np.random.uniform(0., 1.) < self.prob:
            return [im_id, im_shape, im, gt_bbox, gt_class, gt_score]

        height, width, _ = im.shape
        expand_ratio = np.random.uniform(1., self.ratio)
        h = int(height * expand_ratio)
        w = int(width * expand_ratio)
        if not h > height or not w > width:
            return [im_id, im_shape, im, gt_bbox, gt_class, gt_score]
        y = np.random.randint(0, h - height)
        x = np.random.randint(0, w - width)
        canvas = np.ones((h, w, 3), dtype=np.uint8)
        canvas *= np.array(self.fill_value, dtype=np.uint8)
        canvas[y:y + height, x:x + width, :] = im.astype(np.uint8)

        gt_bbox += np.array([x, y, x, y], dtype=np.float32)

        return [im_id, im_shape, canvas, gt_bbox, gt_class, gt_score]


class RandomCrop():
    """Random crop image and bboxes.

    Args:
        aspect_ratio (list): aspect ratio of cropped region.
            in [min, max] format.
        thresholds (list): iou thresholds for decide a valid bbox crop.
        scaling (list): ratio between a cropped region and the original image.
             in [min, max] format.
        num_attempts (int): number of tries before giving up.
        allow_no_crop (bool): allow return without actually cropping them.
        cover_all_box (bool): ensure all bboxes are covered in the final crop.
    """

    def __init__(self,
                 aspect_ratio=[.5, 2.],
                 thresholds=[.0, .1, .3, .5, .7, .9],
                 scaling=[.3, 1.],
                 num_attempts=50,
                 allow_no_crop=True,
                 cover_all_box=False):
        self.aspect_ratio = aspect_ratio
        self.thresholds = thresholds
        self.scaling = scaling
        self.num_attempts = num_attempts
        self.allow_no_crop = allow_no_crop
        self.cover_all_box = cover_all_box

    def __call__(self, im_id, im_shape, im, gt_bbox, gt_class, gt_score):
        if len(gt_bbox) == 0:
            return [im_id, im_shape, im, gt_bbox, gt_class, gt_score]

        # NOTE Original method attempts to generate one candidate for each
        # threshold then randomly sample one from the resulting list.
        # Here a short circuit approach is taken, i.e., randomly choose a
        # threshold and attempt to find a valid crop, and simply return the
        # first one found.
        # The probability is not exactly the same, kinda resembling the
        # "Monty Hall" problem. Actually carrying out the attempts will affect
        # observability (just like opening doors in the "Monty Hall" game).
        thresholds = list(self.thresholds)
        if self.allow_no_crop:
            thresholds.append('no_crop')
        np.random.shuffle(thresholds)

        for thresh in thresholds:
            if thresh == 'no_crop':
                return [im_id, im_shape, im, gt_bbox, gt_class, gt_score]

            h, w, _ = im.shape
            found = False
            for i in range(self.num_attempts):
                scale = np.random.uniform(*self.scaling)
                min_ar, max_ar = self.aspect_ratio
                aspect_ratio = np.random.uniform(
                    max(min_ar, scale**2), min(max_ar, scale**-2))
                crop_h = int(h * scale / np.sqrt(aspect_ratio))
                crop_w = int(w * scale * np.sqrt(aspect_ratio))
                crop_y = np.random.randint(0, h - crop_h)
                crop_x = np.random.randint(0, w - crop_w)
                crop_box = [crop_x, crop_y, crop_x + crop_w, crop_y + crop_h]
                iou = self._iou_matrix(
                    gt_bbox, np.array(
                        [crop_box], dtype=np.float32))
                if iou.max() < thresh:
                    continue

                if self.cover_all_box and iou.min() < thresh:
                    continue

                cropped_box, valid_ids = self._crop_box_with_center_constraint(
                    gt_bbox, np.array(
                        crop_box, dtype=np.float32))
                if valid_ids.size > 0:
                    found = True
                    break

            if found:
                im = self._crop_image(im, crop_box)
                gt_bbox = np.take(cropped_box, valid_ids, axis=0)
                gt_class = np.take(gt_class, valid_ids, axis=0)
                gt_score = np.take(gt_score, valid_ids, axis=0)
                return [im_id, im_shape, im, gt_bbox, gt_class, gt_score]

        return [im_id, im_shape, im, gt_bbox, gt_class, gt_score]

    def _iou_matrix(self, a, b):
        tl_i = np.maximum(a[:, np.newaxis, :2], b[:, :2])
        br_i = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

        area_i = np.prod(br_i - tl_i, axis=2) * (tl_i < br_i).all(axis=2)
        area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
        area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
        area_o = (area_a[:, np.newaxis] + area_b - area_i)
        return area_i / (area_o + 1e-10)

    def _crop_box_with_center_constraint(self, box, crop):
        cropped_box = box.copy()

        cropped_box[:, :2] = np.maximum(box[:, :2], crop[:2])
        cropped_box[:, 2:] = np.minimum(box[:, 2:], crop[2:])
        cropped_box[:, :2] -= crop[:2]
        cropped_box[:, 2:] -= crop[:2]

        centers = (box[:, :2] + box[:, 2:]) / 2
        valid = np.logical_and(crop[:2] <= centers,
                               centers < crop[2:]).all(axis=1)
        valid = np.logical_and(
            valid, (cropped_box[:, :2] < cropped_box[:, 2:]).all(axis=1))

        return cropped_box, np.where(valid)[0]

    def _crop_image(self, img, crop):
        x1, y1, x2, y2 = crop
        return img[y1:y2, x1:x2, :]


class RandomFlip():
    def __init__(self, prob=0.5, is_normalized=False):
        """
        Args:
            prob (float): the probability of flipping image
            is_normalized (bool): whether the bbox scale to [0,1]
        """
        self.prob = prob
        self.is_normalized = is_normalized
        if not (isinstance(self.prob, float) and
                isinstance(self.is_normalized, bool)):
            raise TypeError("{}: input type is invalid.".format(self))

    def __call__(self, im_id, im_shape, im, gt_bbox, gt_class, gt_score):
        """Filp the image and bounding box.
        Operators:
            1. Flip the image numpy.
            2. Transform the bboxes' x coordinates.
              (Must judge whether the coordinates are normalized!)
        """

        if not isinstance(im, np.ndarray):
            raise TypeError("{}: image is not a numpy array.".format(self))
        if len(im.shape) != 3:
            raise ImageError("{}: image is not 3-dimensional.".format(self))
        height, width, _ = im.shape
        if np.random.uniform(0, 1) < self.prob:
            im = im[:, ::-1, :]
            if gt_bbox.shape[0] > 0:
                oldx1 = gt_bbox[:, 0].copy()
                oldx2 = gt_bbox[:, 2].copy()
                if self.is_normalized:
                    gt_bbox[:, 0] = 1 - oldx2
                    gt_bbox[:, 2] = 1 - oldx1
                else:
                    gt_bbox[:, 0] = width - oldx2 - 1
                    gt_bbox[:, 2] = width - oldx1 - 1
                if gt_bbox.shape[0] != 0 and (
                        gt_bbox[:, 2] < gt_bbox[:, 0]).all():
                    m = "{}: invalid box, x2 should be greater than x1".format(
                        self)
                    raise ValueError(m)
        return [im_id, im_shape, im, gt_bbox, gt_class, gt_score]


class NormalizeBox(object):
    """Transform the bounding box's coornidates to [0,1]."""

    def __call__(self, im_id, im_shape, im, gt_bbox, gt_class, gt_score):
        height, width, _ = im.shape
        for i in range(gt_bbox.shape[0]):
            gt_bbox[i][0] = gt_bbox[i][0] / width
            gt_bbox[i][1] = gt_bbox[i][1] / height
            gt_bbox[i][2] = gt_bbox[i][2] / width
            gt_bbox[i][3] = gt_bbox[i][3] / height
        return [im_id, im_shape, im, gt_bbox, gt_class, gt_score]


class PadBox(object):
    def __init__(self, num_max_boxes=50):
        """
        Pad zeros to bboxes if number of bboxes is less than num_max_boxes.
        Args:
            num_max_boxes (int): the max number of bboxes
        """
        self.num_max_boxes = num_max_boxes

    def __call__(self, im_id, im_shape, im, gt_bbox, gt_class, gt_score):
        gt_num = min(self.num_max_boxes, len(gt_bbox))
        num_max = self.num_max_boxes

        pad_bbox = np.zeros((num_max, 4), dtype=np.float32)
        if gt_num > 0:
            pad_bbox[:gt_num, :] = gt_bbox[:gt_num, :]
        gt_bbox = pad_bbox

        pad_class = np.zeros((num_max), dtype=np.int32)
        if gt_num > 0:
            pad_class[:gt_num] = gt_class[:gt_num, 0]
        gt_class = pad_class

        pad_score = np.zeros((num_max), dtype=np.float32)
        if gt_num > 0:
            pad_score[:gt_num] = gt_score[:gt_num, 0]
        gt_score = pad_score
        return [im_id, im_shape, im, gt_bbox, gt_class, gt_score]


class BboxXYXY2XYWH(object):
    """
    Convert bbox XYXY format to XYWH format.
    """

    def __call__(self, im_id, im_shape, im, gt_bbox, gt_class, gt_score):
        gt_bbox[:, 2:4] = gt_bbox[:, 2:4] - gt_bbox[:, :2]
        gt_bbox[:, :2] = gt_bbox[:, :2] + gt_bbox[:, 2:4] / 2.
        return [im_id, im_shape, im, gt_bbox, gt_class, gt_score]


class RandomShape(object):
    """
    Randomly reshape a batch. If random_inter is True, also randomly
    select one an interpolation algorithm [cv2.INTER_NEAREST, cv2.INTER_LINEAR,
    cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]. If random_inter is
    False, use cv2.INTER_NEAREST.

    Args:
        sizes (list): list of int, random choose a size from these
        random_inter (bool): whether to randomly interpolation, defalut true.
    """

    def __init__(self,
                 sizes=[320, 352, 384, 416, 448, 480, 512, 544, 576, 608],
                 random_inter=True):
        self.sizes = sizes
        self.random_inter = random_inter
        self.interps = [
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_AREA,
            cv2.INTER_CUBIC,
            cv2.INTER_LANCZOS4,
        ] if random_inter else []

    def __call__(self, samples):
        shape = np.random.choice(self.sizes)
        method = np.random.choice(self.interps) if self.random_inter \
            else cv2.INTER_NEAREST
        for i in range(len(samples)):
            im = samples[i][2]
            h, w = im.shape[:2]
            scale_x = float(shape) / w
            scale_y = float(shape) / h
            im = cv2.resize(
                im, None, None, fx=scale_x, fy=scale_y, interpolation=method)
            samples[i][2] = im
        return samples


class NormalizeImage(object):
    def __init__(self,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 scale=True,
                 channel_first=True):
        """
        Args:
            mean (list): the pixel mean
            std (list): the pixel variance
            scale (bool):  whether scale image to [0, 1]
            channel_first (bool):  whehter change [h, w, c] to [c, h, w]
        """
        self.mean = mean
        self.std = std
        self.scale = scale
        self.channel_first = channel_first
        if not (isinstance(self.mean, list) and isinstance(self.std, list) and
                isinstance(self.scale, bool)):
            raise TypeError("{}: input type is invalid.".format(self))
        from functools import reduce
        if reduce(lambda x, y: x * y, self.std) == 0:
            raise ValueError('{}: std is invalid!'.format(self))

    def __call__(self, samples):
        """Normalize the image.
        Operators:
            1. (optional) Scale the image to [0,1]
            2. Each pixel minus mean and is divided by std
            3. (optional) permute channel
        """
        for i in range(len(samples)):
            im = samples[i][2]
            im = im.astype(np.float32, copy=False)
            mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
            std = np.array(self.std)[np.newaxis, np.newaxis, :]
            if self.scale:
                im = im / 255.0
            im -= mean
            im /= std
            if self.channel_first:
                im = im.transpose((2, 0, 1))
            samples[i][2] = im
        return samples


def _iou_matrix(a, b):
    tl_i = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    br_i = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])
    area_i = np.prod(br_i - tl_i, axis=2) * (tl_i < br_i).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    area_o = (area_a[:, np.newaxis] + area_b - area_i)
    return area_i / (area_o + 1e-10)


def _crop_box_with_center_constraint(box, crop):
    cropped_box = box.copy()
    cropped_box[:, :2] = np.maximum(box[:, :2], crop[:2])
    cropped_box[:, 2:] = np.minimum(box[:, 2:], crop[2:])
    cropped_box[:, :2] -= crop[:2]
    cropped_box[:, 2:] -= crop[:2]
    centers = (box[:, :2] + box[:, 2:]) / 2
    valid = np.logical_and(crop[:2] <= centers, centers < crop[2:]).all(axis=1)
    valid = np.logical_and(
        valid, (cropped_box[:, :2] < cropped_box[:, 2:]).all(axis=1))
    return cropped_box, np.where(valid)[0]


def random_crop(inputs):
    aspect_ratios = [.5, 2.]
    thresholds = [.0, .1, .3, .5, .7, .9]
    scaling = [.3, 1.]

    img, img_ids, gt_box, gt_label = inputs
    h, w = img.shape[:2]

    if len(gt_box) == 0:
        return inputs

    np.random.shuffle(thresholds)
    for thresh in thresholds:
        found = False
        for i in range(50):
            scale = np.random.uniform(*scaling)
            min_ar, max_ar = aspect_ratios
            ar = np.random.uniform(
                max(min_ar, scale**2), min(max_ar, scale**-2))
            crop_h = int(h * scale / np.sqrt(ar))
            crop_w = int(w * scale * np.sqrt(ar))
            crop_y = np.random.randint(0, h - crop_h)
            crop_x = np.random.randint(0, w - crop_w)
            crop_box = [crop_x, crop_y, crop_x + crop_w, crop_y + crop_h]
            iou = _iou_matrix(gt_box, np.array([crop_box], dtype=np.float32))
            if iou.max() < thresh:
                continue

            cropped_box, valid_ids = _crop_box_with_center_constraint(
                gt_box, np.array(
                    crop_box, dtype=np.float32))
            if valid_ids.size > 0:
                found = True
                break

        if found:
            x1, y1, x2, y2 = crop_box
            img = img[y1:y2, x1:x2, :]
            gt_box = np.take(cropped_box, valid_ids, axis=0)
            gt_label = np.take(gt_label, valid_ids, axis=0)
            return img, img_ids, gt_box, gt_label

        return inputs


class ResizeImage(object):
    def __init__(self, target_size=0, interp=cv2.INTER_CUBIC):
        """
        Rescale image to the specified target size.
        If target_size is list, selected a scale randomly as the specified
        target size.

        Args:
            target_size (int|list): the target size of image's short side,
                multi-scale training is adopted when type is list.
            interp (int): the interpolation method
        """
        self.interp = int(interp)
        if not (isinstance(target_size, int) or isinstance(target_size, list)):
            raise TypeError(
                "Type of target_size is invalid. Must be Integer or List, now is {}".
                format(type(target_size)))
        self.target_size = target_size

    def __call__(self, im_id, im_shape, im, gt_bbox, gt_class, gt_score):
        """ Resize the image numpy.
        """
        if not isinstance(im, np.ndarray):
            raise TypeError("{}: image type is not numpy.".format(self))
        if len(im.shape) != 3:
            raise ImageError('{}: image is not 3-dimensional.'.format(self))
        im_scale_x = float(self.target_size) / float(im.shape[1])
        im_scale_y = float(self.target_size) / float(im.shape[0])
        resize_w = self.target_size
        resize_h = self.target_size

        im = cv2.resize(
            im,
            None,
            None,
            fx=im_scale_x,
            fy=im_scale_y,
            interpolation=self.interp)

        return [im_id, im_shape, im, gt_bbox, gt_class, gt_score]
