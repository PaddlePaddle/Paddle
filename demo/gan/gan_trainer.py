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
import numpy
import cPickle
import sys, os
from PIL import Image

from paddle.trainer.config_parser import parse_config
from paddle.trainer.config_parser import logger
import py_paddle.swig_paddle as api
import matplotlib.pyplot as plt


def plot2DScatter(data, outputfile):
    '''
    Plot the data as a 2D scatter plot and save to outputfile
    data needs to be two dimensinoal
    '''
    x = data[:, 0]
    y = data[:, 1]
    logger.info("The mean vector is %s" % numpy.mean(data, 0))
    logger.info("The std vector is %s" % numpy.std(data, 0))

    heatmap, xedges, yedges = numpy.histogram2d(x, y, bins=50)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.clf()
    plt.scatter(x, y)
    plt.savefig(outputfile, bbox_inches='tight')


def CHECK_EQ(a, b):
    assert a == b, "a=%s, b=%s" % (a, b)


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


def print_parameters(src):
    src_params = [src.getParameter(i) for i in xrange(src.getParameterSize())]

    print "***************"
    for p in src_params:
        print "Name is %s" % p.getName()
        print "value is %s \n" % p.getBuf(api.PARAMETER_VALUE).copyToNumpyArray(
        )


def load_mnist_data(imageFile):
    f = open(imageFile, "rb")
    f.read(16)

    # Define number of samples for train/test
    if "train" in imageFile:
        n = 60000
    else:
        n = 10000

    data = numpy.fromfile(f, 'ubyte', count=n * 28 * 28).reshape((n, 28 * 28))
    data = data / 255.0 * 2.0 - 1.0

    f.close()
    return data.astype('float32')


def load_cifar_data(cifar_path):
    batch_size = 10000
    data = numpy.zeros((5 * batch_size, 32 * 32 * 3), dtype="float32")
    for i in range(1, 6):
        file = cifar_path + "/data_batch_" + str(i)
        fo = open(file, 'rb')
        dict = cPickle.load(fo)
        fo.close()
        data[(i - 1) * batch_size:(i * batch_size), :] = dict["data"]

    data = data / 255.0 * 2.0 - 1.0
    return data


# synthesize 2-D uniform data
def load_uniform_data():
    data = numpy.random.rand(1000000, 2).astype('float32')
    return data


def merge(images, size):
    if images.shape[1] == 28 * 28:
        h, w, c = 28, 28, 1
    else:
        h, w, c = 32, 32, 3
    img = numpy.zeros((h * size[0], w * size[1], c))
    for idx in xrange(size[0] * size[1]):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = \
          ((images[idx, :].reshape((h, w, c), order="F").transpose(1, 0, 2) + 1.0) / 2.0 * 255.0)
    return img.astype('uint8')


def save_images(images, path):
    merged_img = merge(images, [8, 8])
    if merged_img.shape[2] == 1:
        im = Image.fromarray(numpy.squeeze(merged_img)).convert('RGB')
    else:
        im = Image.fromarray(merged_img, mode="RGB")
    im.save(path)


def get_real_samples(batch_size, data_np):
    return data_np[numpy.random.choice(
        data_np.shape[0], batch_size, replace=False), :]


def get_noise(batch_size, noise_dim):
    return numpy.random.normal(size=(batch_size, noise_dim)).astype('float32')


def get_fake_samples(generator_machine, batch_size, noise):
    gen_inputs = api.Arguments.createArguments(1)
    gen_inputs.setSlotValue(0, api.Matrix.createDenseFromNumpy(noise))
    gen_outputs = api.Arguments.createArguments(0)
    generator_machine.forward(gen_inputs, gen_outputs, api.PASS_TEST)
    fake_samples = gen_outputs.getSlotValue(0).copyToNumpyMat()
    return fake_samples


def get_training_loss(training_machine, inputs):
    outputs = api.Arguments.createArguments(0)
    training_machine.forward(inputs, outputs, api.PASS_TEST)
    loss = outputs.getSlotValue(0).copyToNumpyMat()
    return numpy.mean(loss)


def prepare_discriminator_data_batch_pos(batch_size, data_np):
    real_samples = get_real_samples(batch_size, data_np)
    labels = numpy.ones(batch_size, dtype='int32')
    inputs = api.Arguments.createArguments(2)
    inputs.setSlotValue(0, api.Matrix.createDenseFromNumpy(real_samples))
    inputs.setSlotIds(1, api.IVector.createVectorFromNumpy(labels))
    return inputs


def prepare_discriminator_data_batch_neg(generator_machine, batch_size, noise):
    fake_samples = get_fake_samples(generator_machine, batch_size, noise)
    labels = numpy.zeros(batch_size, dtype='int32')
    inputs = api.Arguments.createArguments(2)
    inputs.setSlotValue(0, api.Matrix.createDenseFromNumpy(fake_samples))
    inputs.setSlotIds(1, api.IVector.createVectorFromNumpy(labels))
    return inputs


def prepare_generator_data_batch(batch_size, noise):
    label = numpy.ones(batch_size, dtype='int32')
    inputs = api.Arguments.createArguments(2)
    inputs.setSlotValue(0, api.Matrix.createDenseFromNumpy(noise))
    inputs.setSlotIds(1, api.IVector.createVectorFromNumpy(label))
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
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_source", help="mnist or cifar or uniform")
    parser.add_argument(
        "--use_gpu", default="1", help="1 means use gpu for training")
    parser.add_argument("--gpu_id", default="0", help="the gpu_id parameter")
    args = parser.parse_args()
    data_source = args.data_source
    use_gpu = args.use_gpu
    assert data_source in ["mnist", "cifar", "uniform"]
    assert use_gpu in ["0", "1"]

    if not os.path.exists("./%s_samples/" % data_source):
        os.makedirs("./%s_samples/" % data_source)

    if not os.path.exists("./%s_params/" % data_source):
        os.makedirs("./%s_params/" % data_source)

    api.initPaddle('--use_gpu=' + use_gpu, '--dot_period=10',
                   '--log_period=100', '--gpu_id=' + args.gpu_id,
                   '--save_dir=' + "./%s_params/" % data_source)

    if data_source == "uniform":
        conf = "gan_conf.py"
        num_iter = 10000
    else:
        conf = "gan_conf_image.py"
        num_iter = 1000

    gen_conf = parse_config(conf, "mode=generator_training,data=" + data_source)
    dis_conf = parse_config(conf,
                            "mode=discriminator_training,data=" + data_source)
    generator_conf = parse_config(conf, "mode=generator,data=" + data_source)
    batch_size = dis_conf.opt_config.batch_size
    noise_dim = get_layer_size(gen_conf.model_config, "noise")

    if data_source == "mnist":
        data_np = load_mnist_data("./data/mnist_data/train-images-idx3-ubyte")
    elif data_source == "cifar":
        data_np = load_cifar_data("./data/cifar-10-batches-py/")
    else:
        data_np = load_uniform_data()

    # this creates a gradient machine for discriminator
    dis_training_machine = api.GradientMachine.createFromConfigProto(
        dis_conf.model_config)
    # this create a gradient machine for generator    
    gen_training_machine = api.GradientMachine.createFromConfigProto(
        gen_conf.model_config)

    # generator_machine is used to generate data only, which is used for
    # training discriminator
    logger.info(str(generator_conf.model_config))
    generator_machine = api.GradientMachine.createFromConfigProto(
        generator_conf.model_config)

    dis_trainer = api.Trainer.create(dis_conf, dis_training_machine)

    gen_trainer = api.Trainer.create(gen_conf, gen_training_machine)

    dis_trainer.startTrain()
    gen_trainer.startTrain()

    # Sync parameters between networks (GradientMachine) at the beginning
    copy_shared_parameters(gen_training_machine, dis_training_machine)
    copy_shared_parameters(gen_training_machine, generator_machine)

    # constrain that either discriminator or generator can not be trained
    # consecutively more than MAX_strike times
    curr_train = "dis"
    curr_strike = 0
    MAX_strike = 5

    for train_pass in xrange(100):
        dis_trainer.startTrainPass()
        gen_trainer.startTrainPass()
        for i in xrange(num_iter):
            # Do forward pass in discriminator to get the dis_loss
            noise = get_noise(batch_size, noise_dim)
            data_batch_dis_pos = prepare_discriminator_data_batch_pos(
                batch_size, data_np)
            dis_loss_pos = get_training_loss(dis_training_machine,
                                             data_batch_dis_pos)

            data_batch_dis_neg = prepare_discriminator_data_batch_neg(
                generator_machine, batch_size, noise)
            dis_loss_neg = get_training_loss(dis_training_machine,
                                             data_batch_dis_neg)

            dis_loss = (dis_loss_pos + dis_loss_neg) / 2.0

            # Do forward pass in generator to get the gen_loss
            data_batch_gen = prepare_generator_data_batch(batch_size, noise)
            gen_loss = get_training_loss(gen_training_machine, data_batch_gen)

            if i % 100 == 0:
                print "d_pos_loss is %s     d_neg_loss is %s" % (dis_loss_pos,
                                                                 dis_loss_neg)
                print "d_loss is %s    g_loss is %s" % (dis_loss, gen_loss)

            # Decide which network to train based on the training history
            # And the relative size of the loss        
            if (not (curr_train == "dis" and curr_strike == MAX_strike)) and \
               ((curr_train == "gen" and curr_strike == MAX_strike) or dis_loss > gen_loss):
                if curr_train == "dis":
                    curr_strike += 1
                else:
                    curr_train = "dis"
                    curr_strike = 1
                dis_trainer.trainOneDataBatch(batch_size, data_batch_dis_neg)
                dis_trainer.trainOneDataBatch(batch_size, data_batch_dis_pos)
                copy_shared_parameters(dis_training_machine,
                                       gen_training_machine)

            else:
                if curr_train == "gen":
                    curr_strike += 1
                else:
                    curr_train = "gen"
                    curr_strike = 1
                gen_trainer.trainOneDataBatch(batch_size, data_batch_gen)
                # TODO: add API for paddle to allow true parameter sharing between different GradientMachines 
                # so that we do not need to copy shared parameters. 
                copy_shared_parameters(gen_training_machine,
                                       dis_training_machine)
                copy_shared_parameters(gen_training_machine, generator_machine)

        dis_trainer.finishTrainPass()
        gen_trainer.finishTrainPass()
        # At the end of each pass, save the generated samples/images
        fake_samples = get_fake_samples(generator_machine, batch_size, noise)
        if data_source == "uniform":
            plot2DScatter(fake_samples, "./%s_samples/train_pass%s.png" %
                          (data_source, train_pass))
        else:
            save_images(fake_samples, "./%s_samples/train_pass%s.png" %
                        (data_source, train_pass))
    dis_trainer.finishTrain()
    gen_trainer.finishTrain()


if __name__ == '__main__':
    main()
