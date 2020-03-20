from __future__ import division
from __future__ import print_function

import os
import sys
sys.path.append('../')

import argparse
import contextlib
import time

import numpy as np

import paddle.fluid as fluid

from model import CrossEntropy, Input
from nets import ResNet
from distributed import prepare_context, all_gather, Env, get_local_rank, get_nranks, DistributedBatchSampler
from utils import ImageNetDataset
from metrics import Accuracy
from models.resnet import resnet50
from paddle.fluid.io import BatchSampler, DataLoader

def run(model, loader, mode='train'):
    total_loss = 0
    total_time = 0.0 #AverageMeter()
    local_rank = get_local_rank()
    start = time.time()
    start_time = time.time()
    for idx, batch in enumerate(loader()):
        if not fluid.in_dygraph_mode():
            batch = batch[0]
        
        losses, metrics = getattr(model, mode)(
            batch[0], batch[1])

        if idx > 1:  # skip first two steps
            total_time += time.time() - start
        total_loss += np.sum(losses)
        if idx % 10 == 0 and local_rank == 0:
            print("{:04d}: loss {:0.3f} top1: {:0.3f}% top5: {:0.3f}% time: {:0.3f} samples: {}".format(
                idx, total_loss / (idx + 1), metrics[0][0] * 100, metrics[0][1] * 100, total_time / max(1, (idx - 1)), model._metrics[0].count[0]))
        start = time.time()
    eval_time = time.time() - start_time
    for metric in model._metrics:
        res = metric.accumulate()
        if local_rank == 0:
            print("[EVAL END]: top1: {:0.3f}%, top5: {:0.3f} total samples: {} total time: {:.3f}".format(res[0] * 100, res[1] * 100, model._metrics[0].count[0], eval_time))
        metric.reset()


def main():
    @contextlib.contextmanager
    def null_guard():
        yield

    place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id) \
        if get_nranks() > 1 else fluid.CUDAPlace(0)
    guard = fluid.dygraph.guard(place) if FLAGS.dynamic else null_guard()
    if get_nranks() > 1:
        prepare_context(place)

    if get_nranks() > 1 and not os.path.exists('resnet_checkpoints'):
        os.mkdir('resnet_checkpoints')

    with guard:
        # model = ResNet()
        model = resnet50(pretrained=True)
        
        inputs = [Input([None, 3, 224, 224], 'float32', name='image')]
        labels = [Input([None, 1], 'int64', name='label')]

        if fluid.in_dygraph_mode():
            feed_list = None
        else:
            feed_list = [x.forward() for x in inputs + labels]
            
        val_dataset = ImageNetDataset(os.path.join(FLAGS.data, 'val'), mode='val')
        if get_nranks() > 1:
            distributed_sampler = DistributedBatchSampler(val_dataset, batch_size=FLAGS.batch_size)
            val_loader = DataLoader(val_dataset, batch_sampler=distributed_sampler, places=place, 
                                    feed_list=feed_list, num_workers=4, return_list=True)
        else:
            val_loader = DataLoader(val_dataset, batch_size=FLAGS.batch_size, places=place, 
                                    feed_list=feed_list, num_workers=4, return_list=True)
            
        model.prepare(None, CrossEntropy(), Accuracy(topk=(1, 5)), inputs, labels, val_dataset)

        # model.save('resnet_checkpoints/{:03d}'.format(000))
        if FLAGS.resume is not None:
            model.load(FLAGS.resume)

        run(model, val_loader, mode='eval')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Resnet Training on ImageNet")
    parser.add_argument('data', metavar='DIR', help='path to dataset '
                        '(should have subdirectories named "train" and "val"')
    parser.add_argument(
        "-d", "--dynamic", action='store_true', help="enable dygraph mode")
    parser.add_argument(
        "-e", "--epoch", default=90, type=int, help="number of epoch")
    parser.add_argument(
        '--lr', '--learning-rate', default=0.1, type=float, metavar='LR',
        help='initial learning rate')
    parser.add_argument(
        "-b", "--batch_size", default=4, type=int, help="batch size")
    parser.add_argument(
        "-n", "--num_devices", default=1, type=int, help="number of devices")
    parser.add_argument(
        "-r", "--resume", default=None, type=str,
        help="checkpoint path to resume")
    FLAGS = parser.parse_args()
    assert FLAGS.data, "error: must provide data path"
    main()