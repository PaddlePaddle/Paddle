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

import os
import numpy as np
from optparse import OptionParser
from py_paddle import swig_paddle, DataProviderConverter
from paddle.trainer.PyDataProvider2 import integer_value_sequence
from paddle.trainer.config_parser import parse_config

"""
Usage: run following command to show help message.
  python predict.py -h
"""

class SentimentPrediction():
    def __init__(self, train_conf, dict_file, model_dir=None, label_file = None):
        """
        train_conf: trainer configure.
        dict_file: word dictionary file name.
        model_dir: directory of model.
        """
        self.train_conf = train_conf
        self.dict_file = dict_file
        self.word_dict = {}
        self.dict_dim = self.load_dict()
        self.model_dir = model_dir
        if model_dir is None:
            self.model_dir = os.path.dirname(train_conf)

        self.label = None
        if label_file is not None:
            self.load_label(label_file)

        conf = parse_config(train_conf, "is_predict=1")
        self.network = swig_paddle.GradientMachine.createFromConfigProto(conf.model_config)
        self.network.loadParameters(self.model_dir)
        slots = [integer_value_sequence(self.dict_dim)]
        self.converter = DataProviderConverter(slots)

    def load_dict(self):
        """
        Load dictionary from self.dict_file.
        """
        for line_count, line in enumerate(open(self.dict_file, 'r')):
            self.word_dict[line.strip().split('\t')[0]] = line_count
        return len(self.word_dict)

    def load_label(self, label_file):
        """
        Load label.
        """
        self.label={}
        for v in open(label_file, 'r'):
            self.label[int(v.split('\t')[1])] = v.split('\t')[0]

    def get_data(self, data_file):
        """
        Get input data of paddle format.
        """
        with open(data_file, 'r') as fdata:
            for line in fdata:
                words = line.strip().split()
                word_slot = [self.word_dict[w] for w in words if w in self.word_dict]
                if not word_slot:
                    print "all words are not in dictionary: %s", line
                    continue
                yield [word_slot]

    def predict(self, data_file):
        """
        data_file: file name of input data.
        """
        input = self.converter(self.get_data(data_file))
        output = self.network.forwardTest(input)
        prob = output[0]["value"]
        lab = np.argsort(-prob)
        if self.label is None:
            print("%s: predicting label is %d" % (data_file, lab[0][0]))
        else:
            print("%s: predicting label is %s" % (data_file, self.label[lab[0][0]]))

def option_parser():
    usage = "python predict.py -n config -w model_dir -d dictionary -i input_file "
    parser = OptionParser(usage="usage: %s [options]" % usage)
    parser.add_option("-n", "--tconf", action="store",
                      dest="train_conf", help="network config")
    parser.add_option("-d", "--dict", action="store",
                      dest="dict_file",help="dictionary file")
    parser.add_option("-b", "--label", action="store",
                      dest="label", default=None,
                      help="dictionary file")
    parser.add_option("-i", "--data", action="store",
                      dest="data", help="data file to predict")
    parser.add_option("-w", "--model", action="store",
                      dest="model_path", default=None,
                      help="model path")
    return parser.parse_args()

def main():
    options, args = option_parser()
    train_conf = options.train_conf
    data = options.data
    dict_file = options.dict_file
    model_path = options.model_path
    label = options.label
    swig_paddle.initPaddle("--use_gpu=0")
    predict = SentimentPrediction(train_conf, dict_file, model_path, label)
    predict.predict(data)

if __name__ == '__main__':
    main()
