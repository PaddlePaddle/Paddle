#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import argparse
import sys
import os
import logging
import paddle.fluid as fluid

sys.path.append('../')

from model import set_device, Input
from bmn_metric import BmnMetric
from bmn_model import BMN, BmnLoss
from reader import BmnDataset
from config_utils import *

DATATYPE = 'float32'

logging.root.handlers = []
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("BMN inference.")
    parser.add_argument(
        "-d",
        "--dynamic",
        default=True,
        action='store_true',
        help="enable dygraph mode, only support dynamic mode at present time")
    parser.add_argument(
        '--config_file',
        type=str,
        default='bmn.yaml',
        help='path to config file of model')
    parser.add_argument(
        '--device', type=str, default='GPU', help='default use gpu.')
    parser.add_argument(
        '--weights',
        type=str,
        default="checkpoint/final",
        help='weight path, None to automatically download weights provided by Paddle.'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default="predict_results/",
        help='output dir path, default to use ./predict_results/')
    parser.add_argument(
        '--log_interval',
        type=int,
        default=1,
        help='mini-batch interval to log.')
    args = parser.parse_args()
    return args


# Prediction
def infer_bmn(args):
    # only support dynamic mode at present time
    device = set_device(args.device)
    fluid.enable_dygraph(device) if args.dynamic else None

    config = parse_config(args.config_file)
    infer_cfg = merge_configs(config, 'infer', vars(args))

    if not os.path.isdir(config.INFER.output_path):
        os.makedirs(config.INFER.output_path)
    if not os.path.isdir(config.INFER.result_path):
        os.makedirs(config.INFER.result_path)

    inputs = [
        Input(
            [None, config.MODEL.feat_dim, config.MODEL.tscale],
            'float32',
            name='feat_input')
    ]
    labels = [Input([None, 1], 'int64', name='video_idx')]

    #data
    infer_dataset = BmnDataset(infer_cfg, 'infer')

    model = BMN(config, args.dynamic)
    model.prepare(
        metrics=BmnMetric(
            config, mode='infer'),
        inputs=inputs,
        labels=labels,
        device=device)

    # load checkpoint
    if args.weights:
        assert os.path.exists(
            args.weights +
            ".pdparams"), "Given weight dir {} not exist.".format(args.weights)
    logger.info('load test weights from {}'.format(args.weights))
    model.load(args.weights)

    # here use model.eval instead of model.test, as post process is required in our case
    model.evaluate(
        eval_data=infer_dataset,
        batch_size=infer_cfg.TEST.batch_size,
        num_workers=infer_cfg.TEST.num_workers,
        log_freq=args.log_interval)

    logger.info("[INFER] infer finished")


if __name__ == '__main__':
    args = parse_args()
    infer_bmn(args)
