# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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

import argparse
import random
import numpy as np
import cPickle
import sys, os
from PIL import Image

from paddle.trainer.config_parser import parse_config
from paddle.trainer.config_parser import logger
import py_paddle.swig_paddle as api
import dataloader
import matplotlib.pyplot as plt


def plot_samples(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        plt.subplot(gs[i])
        plt.axis('off')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


def CHECK_EQ(a, b):
    assert a == b, "a=%s, b=%s" % (a, b)


def get_fake_samples(generator_machine, batch_size, noise):
    gen_inputs = api.Arguments.createArguments(1)
    gen_inputs.setSlotValue(0, api.Matrix.createDenseFromNumpy(noise))
    gen_outputs = api.Arguments.createArguments(0)
    generator_machine.forward(gen_inputs, gen_outputs, api.PASS_TEST)
    fake_samples = gen_outputs.getSlotValue(0).copyToNumpyMat()
    return fake_samples


def copy_shared_parameters(src, dst):
    '''
    copy the parameters from src to dst
    :param src: the source of the parameters
    :type src: GradientMachine
    :param dst: the destination of the parameters
    :type dst: GradientMachine
    '''
    src_params = [src.getParameter(i) for i in xrange(src.getParameterSize())]
    src_params = dict([(p.getName(), p) for p in src_params])

    for i in xrange(dst.getParameterSize()):
        dst_param = dst.getParameter(i)
        src_param = src_params.get(dst_param.getName(), None)
        if src_param is None:
            continue
        src_value = src_param.getBuf(api.PARAMETER_VALUE)
        dst_value = dst_param.getBuf(api.PARAMETER_VALUE)
        CHECK_EQ(len(src_value), len(dst_value))
        dst_value.copyFrom(src_value)
        dst_param.setValueUpdated()


def find(iterable, cond):
    for item in iterable:
        if cond(item):
            return item
    return None


def get_layer_size(model_conf, layer_name):
    layer_conf = find(model_conf.layers, lambda x: x.name == layer_name)
    assert layer_conf is not None, "Cannot find '%s' layer" % layer_name
    return layer_conf.size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_gpu", default="1", help="1 means use gpu for training")
    parser.add_argument("--gpu_id", default="0", help="the gpu_id parameter")
    args = parser.parse_args()
    use_gpu = args.use_gpu
    assert use_gpu in ["0", "1"]

    if not os.path.exists("./samples/"):
        os.makedirs("./samples/")

    if not os.path.exists("./params/"):
        os.makedirs("./params/")

    api.initPaddle('--use_gpu=' + use_gpu, '--dot_period=10',
                   '--log_period=1000', '--gpu_id=' + args.gpu_id,
                   '--save_dir=' + "./params/")

    conf = "vae_conf.py"

    trainer_conf = parse_config(conf, "is_generating=False")
    gener_conf = parse_config(conf, "is_generating=True")

    batch_size = trainer_conf.opt_config.batch_size

    noise_dim = get_layer_size(gener_conf.model_config, "noise")

    mnist = dataloader.MNISTloader(batch_size=batch_size)
    mnist.load_data()

    training_machine = api.GradientMachine.createFromConfigProto(
        trainer_conf.model_config)

    generator_machine = api.GradientMachine.createFromConfigProto(
        gener_conf.model_config)

    trainer = api.Trainer.create(trainer_conf, training_machine)

    trainer.startTrain()

    for train_pass in xrange(100):
        trainer.startTrainPass()
        mnist.reset_pointer()
        i = 0
        it = 0
        while mnist.pointer != 0 or i == 0:
            X = mnist.next_batch().astype('float32')

            inputs = api.Arguments.createArguments(1)
            inputs.setSlotValue(0, api.Matrix.createDenseFromNumpy(X))

            trainer.trainOneDataBatch(batch_size, inputs)

            if it % 1000 == 0:

                outputs = api.Arguments.createArguments(0)
                training_machine.forward(inputs, outputs, api.PASS_TEST)
                loss = np.mean(outputs.getSlotValue(0).copyToNumpyMat())
                print "\niter: {}".format(str(it).zfill(3))
                print "VAE loss: {}".format(str(loss).zfill(3))

                #Sync parameters between networks (GradientMachine) at the beginning
                copy_shared_parameters(training_machine, generator_machine)

                z_samples = np.random.randn(batch_size,
                                            noise_dim).astype('float32')
                samples = get_fake_samples(generator_machine, batch_size,
                                           z_samples)

                #Generating the first 16 images for a picture. 
                figure = plot_samples(samples[:16])
                plt.savefig(
                    "./samples/{}_{}.png".format(
                        str(train_pass).zfill(3), str(i).zfill(3)),
                    bbox_inches='tight')
                plt.close(figure)
                i += 1
            it += 1

        trainer.finishTrainPass()
    trainer.finishTrain()


if __name__ == '__main__':
    main()
