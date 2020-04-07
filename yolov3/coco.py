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

import os
import cv2
import numpy as np
from pycocotools.coco import COCO

from paddle.fluid.io import Dataset

import logging
logger = logging.getLogger(__name__)

__all__ = ['COCODataset']


class COCODataset(Dataset):
    """
    Load dataset with MS-COCO format.

    Args:
        dataset_dir (str): root directory for dataset.
        image_dir (str): directory for images.
        anno_path (str): voc annotation file path.
        sample_num (int): number of samples to load, -1 means all.
        use_default_label (bool): whether use the default mapping of
            label to integer index. Default True.
        with_background (bool): whether load background as a class,
            default True.
        transform (callable): callable transform to perform on samples,
            default None.
        mixup (bool): whether return image mixup samples, default False.
        alpha (float): alpha factor of beta distribution to generate
            mixup score, used only when mixup is True, default 1.5
        beta (float): beta factor of beta distribution to generate
            mixup score, used only when mixup is True, default 1.5
    """

    def __init__(self,
                 dataset_dir='',
                 image_dir='',
                 anno_path='',
                 sample_num=-1,
                 with_background=True,
                 transform=None,
                 mixup=False,
                 alpha=1.5,
                 beta=1.5):
        # roidbs is list of dict whose structure is:
        # {
        #     'im_file': im_fname, # image file name
        #     'im_id': im_id, # image id
        #     'h': im_h, # height of image
        #     'w': im_w, # width
        #     'is_crowd': is_crowd,
        #     'gt_class': gt_class,
        #     'gt_bbox': gt_bbox,
        #     'gt_score': gt_score,
        #     'difficult': difficult
        # }

        self._anno_path = os.path.join(dataset_dir, anno_path)
        self._image_dir = os.path.join(dataset_dir, image_dir)
        assert os.path.exists(self._anno_path), \
                "anno_path {} not exists".format(anno_path)
        assert os.path.exists(self._image_dir), \
                "image_dir {} not exists".format(image_dir)

        self._sample_num = sample_num
        self._with_background = with_background
        self._transform = transform
        self._mixup = mixup
        self._alpha = alpha
        self._beta = beta 

        # load in dataset roidbs
        self._load_roidb_and_cname2cid()

    def _load_roidb_and_cname2cid(self):
        assert self._anno_path.endswith('.json'), \
            'invalid coco annotation file: ' + anno_path
        coco = COCO(self._anno_path)
        img_ids = coco.getImgIds()
        cat_ids = coco.getCatIds()
        records = []
        ct = 0

        # when with_background = True, mapping category to classid, like:
        #   background:0, first_class:1, second_class:2, ...
        catid2clsid = dict({
            catid: i + int(self._with_background)
            for i, catid in enumerate(cat_ids)
        })
        cname2cid = dict({
            coco.loadCats(catid)[0]['name']: clsid
            for catid, clsid in catid2clsid.items()
        })

        for img_id in img_ids:
            img_anno = coco.loadImgs(img_id)[0]
            im_fname = img_anno['file_name']
            im_w = float(img_anno['width'])
            im_h = float(img_anno['height'])

            ins_anno_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
            instances = coco.loadAnns(ins_anno_ids)

            bboxes = []
            for inst in instances:
                x, y, box_w, box_h = inst['bbox']
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(im_w - 1, x1 + max(0, box_w - 1))
                y2 = min(im_h - 1, y1 + max(0, box_h - 1))
                if inst['area'] > 0 and x2 >= x1 and y2 >= y1:
                    inst['clean_bbox'] = [x1, y1, x2, y2]
                    bboxes.append(inst)
                else:
                    logger.warn(
                        'Found an invalid bbox in annotations: im_id: {}, '
                        'area: {} x1: {}, y1: {}, x2: {}, y2: {}.'.format(
                            img_id, float(inst['area']), x1, y1, x2, y2))
            num_bbox = len(bboxes)

            gt_bbox = np.zeros((num_bbox, 4), dtype=np.float32)
            gt_class = np.zeros((num_bbox, 1), dtype=np.int32)
            gt_score = np.ones((num_bbox, 1), dtype=np.float32)
            is_crowd = np.zeros((num_bbox, 1), dtype=np.int32)
            difficult = np.zeros((num_bbox, 1), dtype=np.int32)
            gt_poly = [None] * num_bbox

            for i, box in enumerate(bboxes):
                catid = box['category_id']
                gt_class[i][0] = catid2clsid[catid]
                gt_bbox[i, :] = box['clean_bbox']
                is_crowd[i][0] = box['iscrowd']
                if 'segmentation' in box:
                    gt_poly[i] = box['segmentation']

            im_fname = os.path.join(self._image_dir,
                                    im_fname) if self._image_dir else im_fname
            coco_rec = {
                'im_file': im_fname,
                'im_id': np.array([img_id]),
                'h': im_h,
                'w': im_w,
                'is_crowd': is_crowd,
                'gt_class': gt_class,
                'gt_bbox': gt_bbox,
                'gt_score': gt_score,
                'gt_poly': gt_poly,
            }

            records.append(coco_rec)
            ct += 1
            if self._sample_num > 0 and ct >= self._sample_num:
                break
        assert len(records) > 0, 'not found any coco record in %s' % (self._anno_path)
        logger.info('{} samples in file {}'.format(ct, self._anno_path))
        self._roidbs, self._cname2cid = records, cname2cid

    @property
    def num_classes(self):
        return len(self._cname2cid)

    def __len__(self):
        return len(self._roidbs)

    def _getitem_by_index(self, idx):
        roidb = self._roidbs[idx]
        with open(roidb['im_file'], 'rb') as f:
            data = np.frombuffer(f.read(), dtype='uint8')
            im = cv2.imdecode(data, 1)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_info = np.array([roidb['im_id'][0], roidb['h'], roidb['w']], dtype='int32')
        gt_bbox = roidb['gt_bbox']
        gt_class = roidb['gt_class']
        gt_score = roidb['gt_score']
        return im_info, im, gt_bbox, gt_class, gt_score

    def __getitem__(self, idx):
        im_info, im, gt_bbox, gt_class, gt_score = self._getitem_by_index(idx)

        if self._mixup:
            mixup_idx = idx + np.random.randint(1, self.__len__())
            mixup_idx %= self.__len__()
            _, mixup_im, mixup_bbox, mixup_class, _ = \
                            self._getitem_by_index(mixup_idx)
            
            im, gt_bbox, gt_class, gt_score = \
                    self._mixup_image(im, gt_bbox, gt_class, mixup_im,
                                      mixup_bbox, mixup_class)

        if self._transform:
            im_info, im, gt_bbox, gt_class, gt_score = \
                    self._transform(im_info, im, gt_bbox, gt_class, gt_score)

        return [im_info, im, gt_bbox, gt_class, gt_score]

    def _mixup_image(self, img1, bbox1, class1, img2, bbox2, class2):
        factor = np.random.beta(self._alpha, self._beta)
        factor = max(0.0, min(1.0, factor))
        if factor >= 1.0:
            return img1, bbox1, class1, np.ones_like(class1, dtype="float32")
        if factor <= 0.0:
            return img2, bbox2, class2, np.ones_like(class2, dtype="float32")

        h = max(img1.shape[0], img2.shape[0])
        w = max(img1.shape[1], img2.shape[1])
        img = np.zeros((h, w, img1.shape[2]), 'float32')
        img[:img1.shape[0], :img1.shape[1], :] = \
            img1.astype('float32') * factor
        img[:img2.shape[0], :img2.shape[1], :] += \
            img2.astype('float32') * (1.0 - factor)

        gt_bbox = np.concatenate((bbox1, bbox2), axis=0)
        gt_class = np.concatenate((class1, class2), axis=0)

        score1 = np.ones_like(class1, dtype="float32") * factor
        score2 = np.ones_like(class2, dtype="float32") * (1.0 - factor)
        gt_score = np.concatenate((score1, score2), axis=0)

        return img, gt_bbox, gt_class, gt_score
    
    @property
    def mixup(self):
        return self._mixup

    @mixup.setter
    def mixup(self, value):
        if not isinstance(value, bool):
            raise ValueError("mixup should be a boolean number")
        logger.info("{} set mixup to {}".format(self, value))
        self._mixup = value

def pascalvoc_label(with_background=True):
    labels_map = {
        'aeroplane': 1,
        'bicycle': 2,
        'bird': 3,
        'boat': 4,
        'bottle': 5,
        'bus': 6,
        'car': 7,
        'cat': 8,
        'chair': 9,
        'cow': 10,
        'diningtable': 11,
        'dog': 12,
        'horse': 13,
        'motorbike': 14,
        'person': 15,
        'pottedplant': 16,
        'sheep': 17,
        'sofa': 18,
        'train': 19,
        'tvmonitor': 20
    }
    if not with_background:
        labels_map = {k: v - 1 for k, v in labels_map.items()}
    return labels_map
