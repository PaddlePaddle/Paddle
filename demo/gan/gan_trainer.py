# Copyright (c) 2016 Baidu, Inc. All Rights Reserved
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
import itertools
import random
import numpy

from paddle.trainer.config_parser import parse_config
from paddle.trainer.config_parser import logger
import py_paddle.swig_paddle as api
from py_paddle import DataProviderConverter


def CHECK_EQ(a, b):
    assert a == b, "a=%s, b=%s" % (a, b)


def copy_shared_parameters(src, dst):
    src_params = [src.getParameter(i)
               for i in xrange(src.getParameterSize())]
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


def get_real_samples(batch_size, sample_dim):
    return numpy.random.rand(batch_size, sample_dim).astype('float32')


def prepare_discriminator_data_batch(
        generator_machine, batch_size, noise_dim, sample_dim):
    gen_inputs = prepare_generator_data_batch(batch_size / 2, noise_dim)
    gen_inputs.resize(1)
    gen_outputs = api.Arguments.createArguments(0)
    generator_machine.forward(gen_inputs, gen_outputs, api.PASS_TEST)
    fake_samples = gen_outputs.getSlotValue(0).copyToNumpyMat()
    real_samples = get_real_samples(batch_size / 2, sample_dim)
    all_samples = numpy.concatenate((fake_samples, real_samples), 0)
    all_labels = numpy.concatenate(
        (numpy.zeros(batch_size / 2, dtype='int32'),
         numpy.ones(batch_size / 2, dtype='int32')), 0)
    inputs = api.Arguments.createArguments(2)
    inputs.setSlotValue(0, api.Matrix.createCpuDenseFromNumpy(all_samples))
    inputs.setSlotIds(1, api.IVector.createCpuVectorFromNumpy(all_labels))
    return inputs


def prepare_generator_data_batch(batch_size, dim):
    noise = numpy.random.normal(size=(batch_size, dim)).astype('float32')
    label = numpy.ones(batch_size, dtype='int32')
    inputs = api.Arguments.createArguments(2)
    inputs.setSlotValue(0, api.Matrix.createCpuDenseFromNumpy(noise))
    inputs.setSlotIds(1, api.IVector.createCpuVectorFromNumpy(label))
    return inputs


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
    api.initPaddle('--use_gpu=0', '--dot_period=100', '--log_period=10000')
    gen_conf = parse_config("gan_conf.py", "mode=generator_training")
    dis_conf = parse_config("gan_conf.py", "mode=discriminator_training")
    generator_conf = parse_config("gan_conf.py", "mode=generator")
    batch_size = dis_conf.opt_config.batch_size
    noise_dim = get_layer_size(gen_conf.model_config, "noise")
    sample_dim = get_layer_size(dis_conf.model_config, "sample")

    # this create a gradient machine for discriminator
    dis_training_machine = api.GradientMachine.createFromConfigProto(
        dis_conf.model_config)

    gen_training_machine = api.GradientMachine.createFromConfigProto(
        gen_conf.model_config)

    # generator_machine is used to generate data only, which is used for
    # training discrinator
    logger.info(str(generator_conf.model_config))
    generator_machine = api.GradientMachine.createFromConfigProto(
        generator_conf.model_config)

    dis_trainer = api.Trainer.create(
        dis_conf, dis_training_machine)

    gen_trainer = api.Trainer.create(
        gen_conf, gen_training_machine)

    dis_trainer.startTrain()
    gen_trainer.startTrain()
    for train_pass in xrange(10):
        dis_trainer.startTrainPass()
        gen_trainer.startTrainPass()
        for i in xrange(100000):
            copy_shared_parameters(gen_training_machine, generator_machine)
            copy_shared_parameters(gen_training_machine, dis_training_machine)
            data_batch = prepare_discriminator_data_batch(
                generator_machine, batch_size, noise_dim, sample_dim)
            dis_trainer.trainOneDataBatch(batch_size, data_batch)

            copy_shared_parameters(dis_training_machine, gen_training_machine)
            data_batch = prepare_generator_data_batch(
                batch_size, noise_dim)
            gen_trainer.trainOneDataBatch(batch_size, data_batch)
        dis_trainer.finishTrainPass()
        gen_trainer.finishTrainPass()
    dis_trainer.finishTrain()
    gen_trainer.finishTrain()

if __name__ == '__main__':
    main()
