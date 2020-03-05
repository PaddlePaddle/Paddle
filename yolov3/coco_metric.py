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

import sys
import json
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

from metrics import Metric

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

__all__ = ['COCOMetric']


OUTFILE = './bbox.json'


# considered to change to a callback later
class COCOMetric(Metric):
    """
    Metrci for MS-COCO dataset, only support update with batch
    size as 1.

    Args:
        anno_path(str): path to COCO annotation json file
        with_background(bool): whether load category id with
                               background as 0, default True
    """

    def __init__(self, anno_path, with_background=True, **kwargs):
        super(COCOMetric, self).__init__(**kwargs)
        self.anno_path = anno_path
        self.with_background = with_background
        self.bbox_results = []

        self.coco_gt = COCO(anno_path)
        cat_ids = self.coco_gt.getCatIds()
	self.clsid2catid = dict(
	    {i + int(with_background): catid
	     for i, catid in enumerate(cat_ids)})

    def update(self, preds, *args, **kwargs):
        im_ids, bboxes = preds
        assert im_ids.shape[0] == 1, \
            "COCOMetric can only update with batch size = 1"
        if bboxes.shape[1] != 6:
            # no bbox detected in this batch
            return

        im_id = int(im_ids)
        for i in range(bboxes.shape[0]):
            dt = bboxes[i, :]
            clsid, score, xmin, ymin, xmax, ymax = dt.tolist()
            catid = (self.clsid2catid[int(clsid)])

            w = xmax - xmin + 1
            h = ymax - ymin + 1
            bbox = [xmin, ymin, w, h]
            coco_res = {
                'image_id': im_id,
                'category_id': catid,
                'bbox': bbox,
                'score': score
            }
            self.bbox_results.append(coco_res)

    def reset(self):
        self.bbox_results = []

    def accumulate(self):
	if len(self.bbox_results) == 0:
	    logger.warning("The number of valid bbox detected is zero.\n \
		Please use reasonable model and check input data.\n \
		stop COCOMetric accumulate!")
	    return [0.0]
	with open(OUTFILE, 'w') as f:
	    json.dump(self.bbox_results, f)

	map_stats = self.cocoapi_eval(OUTFILE, 'bbox', coco_gt=self.coco_gt)
	# flush coco evaluation result
	sys.stdout.flush()
        self.result = map_stats[0]
	return self.result

    def cocoapi_eval(self, jsonfile, style, coco_gt=None, anno_file=None):
	assert coco_gt != None or anno_file != None

	if coco_gt == None:
	    coco_gt = COCO(anno_file)
	logger.info("Start evaluate...")
	coco_dt = coco_gt.loadRes(jsonfile)
	coco_eval = COCOeval(coco_gt, coco_dt, style)
	coco_eval.evaluate()
	coco_eval.accumulate()
	coco_eval.summarize()
	return coco_eval.stats

