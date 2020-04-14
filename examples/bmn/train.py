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

import paddle.fluid as fluid
import argparse
import logging
import sys
import os

from hapi.model import set_device, Input
from hapi.vision.models import bmn, BmnLoss
from reader import BmnDataset
from config_utils import *

DATATYPE = 'float32'

logging.root.handlers = []
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("Paddle high level api of BMN.")
    parser.add_argument(
        "-d",
        "--dynamic",
        default=True,
        action='store_true',
        help="enable dygraph mode")
    parser.add_argument(
        '--config_file',
        type=str,
        default='bmn.yaml',
        help='path to config file of model')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='training batch size. None to use config file setting.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='learning rate use for training. None to use config file setting.')
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='filename to resume training based on previous checkpoints. '
        'None for not resuming any checkpoints.')
    parser.add_argument(
        '--device',
        type=str,
        default='gpu',
        help='gpu or cpu, default use gpu.')
    parser.add_argument(
        '--epoch',
        type=int,
        default=9,
        help='epoch number, 0 for read from config file')
    parser.add_argument(
        '--valid_interval',
        type=int,
        default=1,
        help='validation epoch interval, 0 for no validation.')
    parser.add_argument(
        '--save_dir',
        type=str,
        default="checkpoint",
        help='path to save train snapshoot')
    parser.add_argument(
        '--log_interval',
        type=int,
        default=10,
        help='mini-batch interval to log.')
    args = parser.parse_args()
    return args


# Optimizer
def optimizer(cfg, parameter_list):
    bd = [cfg.TRAIN.lr_decay_iter]
    base_lr = cfg.TRAIN.learning_rate
    lr_decay = cfg.TRAIN.learning_rate_decay
    l2_weight_decay = cfg.TRAIN.l2_weight_decay
    lr = [base_lr, base_lr * lr_decay]
    optimizer = fluid.optimizer.Adam(
        fluid.layers.piecewise_decay(
            boundaries=bd, values=lr),
        parameter_list=parameter_list,
        regularization=fluid.regularizer.L2DecayRegularizer(
            regularization_coeff=l2_weight_decay))
    return optimizer


# TRAIN
def train_bmn(args):
    device = set_device(args.device)
    fluid.enable_dygraph(device) if args.dynamic else None

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    config = parse_config(args.config_file)
    train_cfg = merge_configs(config, 'train', vars(args))
    val_cfg = merge_configs(config, 'valid', vars(args))

    inputs = [
        Input(
            [None, config.MODEL.feat_dim, config.MODEL.tscale],
            'float32',
            name='feat_input')
    ]
    gt_iou_map = Input(
        [None, config.MODEL.dscale, config.MODEL.tscale],
        'float32',
        name='gt_iou_map')
    gt_start = Input([None, config.MODEL.tscale], 'float32', name='gt_start')
    gt_end = Input([None, config.MODEL.tscale], 'float32', name='gt_end')
    labels = [gt_iou_map, gt_start, gt_end]

    # data
    train_dataset = BmnDataset(train_cfg, 'train')
    val_dataset = BmnDataset(val_cfg, 'valid')

    # model
    model = bmn(config, args.dynamic, pretrained=False)
    optim = optimizer(config, parameter_list=model.parameters())
    model.prepare(
        optimizer=optim,
        loss_function=BmnLoss(config),
        inputs=inputs,
        labels=labels,
        device=device)

    # if resume weights is given, load resume weights directly
    if args.resume is not None:
        model.load(args.resume)

    model.fit(train_data=train_dataset,
              eval_data=val_dataset,
              batch_size=train_cfg.TRAIN.batch_size,
              epochs=args.epoch,
              eval_freq=args.valid_interval,
              log_freq=args.log_interval,
              save_dir=args.save_dir,
              shuffle=train_cfg.TRAIN.use_shuffle,
              num_workers=train_cfg.TRAIN.num_workers,
              drop_last=True)


if __name__ == "__main__":
    args = parse_args()
    train_bmn(args)
