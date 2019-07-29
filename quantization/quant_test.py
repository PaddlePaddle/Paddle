# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import argparse
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable

from models.resnet import ResNet
from quantization_toolbox.asymmetric_quantization import asymmetric_quantization

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--batch-size', type=int, default=32, metavar='N')
parser.add_argument('--use_data_parallel', type=int, default=0, metavar='N')

args = parser.parse_args()
batch_size = args.batch_size


def eval(model, data):

    model.eval()
    total_loss = 0.0
    total_acc1 = 0.0
    total_acc5 = 0.0
    total_sample = 0
    for batch_id, data in enumerate(data()):
        dy_x_data = np.array(
            [x[0].reshape(3, 224, 224) for x in data]).astype('float32')
        if len(np.array([x[1] for x in data]).astype('int64')) != batch_size:
            continue
        y_data = np.array([x[1] for x in data]).astype('int64').reshape(
            batch_size, 1)

        img = to_variable(dy_x_data)
        label = to_variable(y_data)
        label._stop_gradient = True

        out = model(img)

        acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
        acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)

        total_acc1 += acc_top1.numpy()
        total_acc5 += acc_top5.numpy()
        total_sample += 1

        if batch_id % 10 == 0:
            print("test | batch step %d, loss %0.3f acc1 %0.3f acc5 %0.3f" % \
                  ( batch_id, total_loss / total_sample, \
                   total_acc1 / total_sample, total_acc5 / total_sample))
    print("final eval loss %0.3f acc1 %0.3f acc5 %0.3f" % \
          (total_loss / total_sample, \
           total_acc1 / total_sample, total_acc5 / total_sample))


def quant_net(quant_method):
    place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id) \
        if args.use_data_parallel else fluid.CUDAPlace(0)
    with fluid.dygraph.guard(place):
        resnet_tgt = ResNet("resnet")
        ori_dict2, _ = fluid.dygraph.load_persistables("resnet_params")
        target_dict = quant_method.quantization(ori_dict2)
        resnet_tgt.load_dict(target_dict)
        test_reader = paddle.batch(
            paddle.dataset.flowers.test(use_xmap=False), batch_size=batch_size)
        resnet_tgt.eval()
        eval(resnet_tgt, test_reader)


if __name__ == '__main__':

    quant_method = asymmetric_quantization(256)
    quant_net(quant_method)
