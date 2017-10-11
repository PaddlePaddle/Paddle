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

import os
import sys
import cPickle
import logging
from PIL import Image
import numpy as np
from optparse import OptionParser

import paddle.utils.image_util as image_util

from py_paddle import swig_paddle, DataProviderConverter
from paddle.trainer.PyDataProvider2 import dense_vector
from paddle.trainer.config_parser import parse_config

logging.basicConfig(
    format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s')
logging.getLogger().setLevel(logging.INFO)


class ImageClassifier():
    def __init__(self,
                 train_conf,
                 model_dir=None,
                 resize_dim=256,
                 crop_dim=224,
                 use_gpu=True,
                 mean_file=None,
                 output_layer=None,
                 oversample=False,
                 is_color=True):
        """
        train_conf: network configure.
        model_dir: string, directory of model.
        resize_dim: int, resized image size.
        crop_dim: int, crop size.
        mean_file: string, image mean file.
        oversample: bool, oversample means multiple crops, namely five
                    patches (the four corner patches and the center
                    patch) as well as their horizontal reflections,
                    ten crops in all.
        """
        self.train_conf = train_conf
        self.model_dir = model_dir
        if model_dir is None:
            self.model_dir = os.path.dirname(train_conf)

        self.resize_dim = resize_dim
        self.crop_dims = [crop_dim, crop_dim]
        self.oversample = oversample
        self.is_color = is_color

        self.output_layer = output_layer
        if self.output_layer:
            assert isinstance(self.output_layer, basestring)
            self.output_layer = self.output_layer.split(",")

        self.transformer = image_util.ImageTransformer(is_color=is_color)
        self.transformer.set_transpose((2, 0, 1))
        self.transformer.set_channel_swap((2, 1, 0))

        self.mean_file = mean_file
        if self.mean_file is not None:
            mean = np.load(self.mean_file)['data_mean']
            mean = mean.reshape(3, self.crop_dims[0], self.crop_dims[1])
            self.transformer.set_mean(mean)  # mean pixel
        else:
            # if you use three mean value, set like:
            # this three mean value is calculated from ImageNet.
            self.transformer.set_mean(np.array([103.939, 116.779, 123.68]))

        conf_args = "is_test=1,use_gpu=%d,is_predict=1" % (int(use_gpu))
        conf = parse_config(train_conf, conf_args)
        swig_paddle.initPaddle("--use_gpu=%d" % (int(use_gpu)))
        self.network = swig_paddle.GradientMachine.createFromConfigProto(
            conf.model_config)
        assert isinstance(self.network, swig_paddle.GradientMachine)
        self.network.loadParameters(self.model_dir)

        data_size = 3 * self.crop_dims[0] * self.crop_dims[1]
        slots = [dense_vector(data_size)]
        self.converter = DataProviderConverter(slots)

    def get_data(self, img_path):
        """
        1. load image from img_path.
        2. resize or oversampling.
        3. transformer data: transpose, channel swap, sub mean.
        return K x H x W ndarray.

        img_path: image path.
        """
        image = image_util.load_image(img_path, self.is_color)
        # Another way to extract oversampled features is that
        # cropping and averaging from large feature map which is
        # calculated by large size of image.
        # This way reduces the computation.
        if self.oversample:
            # image_util.resize_image: short side is self.resize_dim
            image = image_util.resize_image(image, self.resize_dim)
            image = np.array(image)
            input = np.zeros(
                (1, image.shape[0], image.shape[1], 3), dtype=np.float32)
            input[0] = image.astype(np.float32)
            input = image_util.oversample(input, self.crop_dims)
        else:
            image = image.resize(self.crop_dims, Image.ANTIALIAS)
            input = np.zeros(
                (1, self.crop_dims[0], self.crop_dims[1], 3), dtype=np.float32)
            input[0] = np.array(image).astype(np.float32)

        data_in = []
        for img in input:
            img = self.transformer.transformer(img).flatten()
            data_in.append([img.tolist()])
        # paddle input: [[[]],[[]],...], [[]] is one sample.
        return data_in

    def forward(self, input_data):
        """
        return output arguments which are the Outputs() in network configure.

        input_data: py_paddle input data.
        call forward.
        """
        in_arg = self.converter(input_data)
        return self.network.forwardTest(in_arg)

    def forward(self, data, output_layer):
        """
        return output arguments which are the Outputs() in network configure.

        input_data: py_paddle input data.
        call forward.
        """
        input = self.converter(data)
        self.network.forwardTest(input)
        output = self.network.getLayerOutputs(output_layer)
        res = {}
        if isinstance(output_layer, basestring):
            output_layer = [output_layer]
        for name in output_layer:
            # For oversampling, average predictions across crops.
            # If not, the shape of output[name]: (1, class_number),
            # the mean is also applicable.
            res[name] = output[name]['value'].mean(0)

        return res

    def predict(self, data_file):
        """
        call forward and predicting.

        data_file: input image list.
        """
        image_files = open(data_file, 'rb').readlines()
        results = {}
        if self.output_layer is None:
            self.output_layer = ["output"]
        for line in image_files:
            image = line.split()[0]
            data = self.get_data(image)
            prob = self.forward(data, self.output_layer)
            lab = np.argsort(-prob[self.output_layer[0]])
            results[image] = lab[0]
            logging.info("Label of %s is: %d", image, lab[0])
        return results

    def extract(self, data_file, output_dir, batch_size=10000):
        """
        extract and save features of output layers, which are
        specify in Outputs() in network configure.

        data_file: file name of input data.
        output_dir: saved directory of extracted features.
        batch_size: sample number of one batch file.
        """
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        sample_num = 0
        batch_num = 0
        image_feature = {}
        image_files = open(data_file, 'rb').readlines()
        for idx, line in enumerate(image_files):
            image = line.split()[0]
            data = self.get_data(image)
            feature = self.forward(data, self.output_layer)
            # save extracted features
            file_name = image.split("/")[-1]
            image_feature[file_name] = feature
            sample_num += 1
            if sample_num == batch_size:
                batch_name = os.path.join(output_dir, 'batch_%d' % (batch_num))
                self.save_file(image_feature, batch_name)
                logging.info('Finish batch %d', batch_num)
                batch_num += 1
                sample_num = 0
                image_feature = {}
            if idx % 1000 == 0:
                logging.info('%d/%d, %s', idx, len(image_files), file_name)
        if sample_num > 0:
            batch_name = os.path.join(output_dir, 'batch_%d' % (batch_num))
            self.save_file(image_feature, batch_name)
            logging.info('Finish batch %d', batch_num)
        logging.info('Done: make image feature batch')

    def save_file(self, data, file):
        of = open(file, 'wb')
        cPickle.dump(data, of, protocol=cPickle.HIGHEST_PROTOCOL)


def option_parser():
    """
    Main entry for predciting
    """
    usage = "%prog -c config -i data_list -w model_dir [options]"
    parser = OptionParser(usage="usage: %s" % usage)
    parser.add_option(
        "-j",
        "--job",
        action="store",
        dest="job_type",
        help="job type: predict, extract\
                            predict: predicting,\
                            extract: extract features")
    parser.add_option(
        "-c",
        "--conf",
        action="store",
        dest="train_conf",
        help="network config")
    parser.add_option(
        "-i", "--data", action="store", dest="data_file", help="image list")
    parser.add_option(
        "-w",
        "--model",
        action="store",
        dest="model_path",
        default=None,
        help="model path")
    parser.add_option(
        "-g",
        "--use_gpu",
        action="store",
        dest="use_gpu",
        default=True,
        help="Whether to use gpu mode.")
    parser.add_option(
        "-o",
        "--output_dir",
        action="store",
        dest="output_dir",
        default="output",
        help="output path")
    parser.add_option(
        "-m",
        "--mean",
        action="store",
        dest="mean",
        default=None,
        help="mean file.")
    parser.add_option(
        "-p",
        "--multi_crop",
        action="store_true",
        dest="multi_crop",
        default=False,
        help="Wether to use multiple crops on image.")
    parser.add_option("-l", "--output_layer", action="store",
                      dest="output_layer", default=None,
                      help="--job=extract, specify layers to extract "\
                           "features, --job=predict, specify layer of "
                           "classification probability, output in resnet.py.")
    return parser.parse_args()


def main():
    """
    1. parse input arguments.
    2. predicting or extract features according job type.
    """
    options, args = option_parser()
    obj = ImageClassifier(
        options.train_conf,
        options.model_path,
        use_gpu=options.use_gpu,
        mean_file=options.mean,
        output_layer=options.output_layer,
        oversample=options.multi_crop)
    if options.job_type == "predict":
        obj.predict(options.data_file)

    elif options.job_type == "extract":
        obj.extract(options.data_file, options.output_dir)


if __name__ == '__main__':
    main()
