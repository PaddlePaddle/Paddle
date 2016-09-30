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

from paddle.trainer.config_parser import parse_config
from py_paddle import swig_paddle, DataProviderConverter
from paddle.trainer.PyDataProvider2 \
    import integer_value, integer_value_sequence, sparse_binary_vector

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data",
                        type=str, required=False, help="train data file")
    parser.add_argument("--test_data", type=str, help="test data file")
    parser.add_argument("--config",
                        type=str, required=True, help="config file name")
    parser.add_argument("--dict_file", required=True, help="dictionary file")
    parser.add_argument("--seq",
                        default=1, type=int, help="whether use sequence training")
    return parser.parse_args()

UNK_IDX = 0

def load_data(file_name, word_dict):
    with open(file_name, 'r') as f:
        for line in f:
            label, comment = line.strip().split('\t')
            words = comment.split()
            word_slot = [word_dict.get(w, UNK_IDX) for w in words]
            yield word_slot, int(label)

def load_dict(dict_file):
    word_dict = dict()
    with open(dict_file, 'r') as f:
        for i, line in enumerate(f):
            w = line.strip().split()[0]
            word_dict[w] = i
    return word_dict

def main():
    options = parse_arguments()
    print 'seq=%s' % options.seq
    swig_paddle.initPaddle("--use_gpu=1", "--v=5", "--trainer_count=2")

    word_dict = load_dict(options.dict_file)
    train_dataset = list(load_data(options.train_data, word_dict))
    if options.test_data:
        test_dataset = list(load_data(options.test_data, word_dict))
    else:
        test_dataset = None

    trainer_config = parse_config(options.config,
                                  "dict_file=%s" % options.dict_file)
    # No need to have data provider for trainer
    trainer_config.ClearField('data_config')
    trainer_config.ClearField('test_data_config')

    print trainer_config.model_config
    model = swig_paddle.GradientMachine.createFromConfigProto(
        trainer_config.model_config)
    trainer = swig_paddle.Trainer.create(trainer_config, model)

    input_types = [
        integer_value_sequence(len(word_dict)) if options.seq
            else sparse_binary_vector(len(word_dict)),
        integer_value(2)]
    converter = DataProviderConverter(input_types)

    batch_size = trainer_config.opt_config.batch_size
    trainer.startTrain()
    for train_pass in xrange(2):
        trainer.startTrainPass()
        random.shuffle(train_dataset)
        for pos in xrange(0, len(train_dataset)):
            batch = itertools.islice(train_dataset, pos, pos + batch_size)
            size = min(batch_size, len(train_dataset) - pos)
            trainer.trainOneDataBatch(size, converter(batch))
        trainer.finishTrainPass()
        if test_dataset:
            trainer.startTestPeriod();
            for pos in xrange(0, len(test_dataset)):
                batch = itertools.islice(test_dataset, pos, pos + batch_size)
                size = min(batch_size, len(test_dataset) - pos)
                trainer.testOneDataBatch(size, converter(batch))
            trainer.finishTestPeriod()
    trainer.finishTrain()

if __name__ == '__main__':
    main()
