# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import logging

logging.basicConfig()
logger = logging.getLogger("paddle")
logger.setLevel(logging.INFO)


class TaskMode:
    TRAIN_MODE = 0
    TEST_MODE = 1
    INFER_MODE = 2

    def __init__(self, mode):
        self.mode = mode

    def is_train(self):
        return self.mode == self.TRAIN_MODE

    def is_test(self):
        return self.mode == self.TEST_MODE

    def is_infer(self):
        return self.mode == self.INFER_MODE

    @staticmethod
    def create_train():
        return TaskMode(TaskMode.TRAIN_MODE)

    @staticmethod
    def create_test():
        return TaskMode(TaskMode.TEST_MODE)

    @staticmethod
    def create_infer():
        return TaskMode(TaskMode.INFER_MODE)


class ModelType:
    CLASSIFICATION = 0
    REGRESSION = 1

    def __init__(self, mode):
        self.mode = mode

    def is_classification(self):
        return self.mode == self.CLASSIFICATION

    def is_regression(self):
        return self.mode == self.REGRESSION

    @staticmethod
    def create_classification():
        return ModelType(ModelType.CLASSIFICATION)

    @staticmethod
    def create_regression():
        return ModelType(ModelType.REGRESSION)


def load_dnn_input_record(sent):
    return map(int, sent.split())


def load_lr_input_record(sent):
    res = []
    for _ in [x.split(':') for x in sent.split()]:
        res.append(int(_[0]))
    return res


feeding_index = {'dnn_input': 0, 'lr_input': 1, 'click': 2}


class Dataset(object):
    def train(self, path):
        '''
        Load trainset.
        '''
        logger.info("load trainset from %s" % path)
        mode = TaskMode.create_train()
        return self._parse_creator(path, mode)

    def test(self, path):
        '''
        Load testset.
        '''
        logger.info("load testset from %s" % path)
        mode = TaskMode.create_test()
        return self._parse_creator(path, mode)

    def infer(self, path):
        '''
        Load infer set.
        '''
        logger.info("load inferset from %s" % path)
        mode = TaskMode.create_infer()
        return self._parse_creator(path, mode)

    def _parse_creator(self, path, mode):
        '''
        Parse dataset.
        '''

        def _parse():
            with open(path) as f:
                for line_id, line in enumerate(f):
                    fs = line.strip().split('\t')
                    dnn_input = load_dnn_input_record(fs[0])
                    lr_input = load_lr_input_record(fs[1])
                    if not mode.is_infer():
                        click = int(fs[2])
                        yield [dnn_input, lr_input, click]
                    else:
                        yield [dnn_input, lr_input]

        return _parse


def load_data_meta(path):
    '''
    load data meta info from path, return (dnn_input_dim, lr_input_dim)
    '''
    with open(path) as f:
        lines = f.read().split('\n')
        err_info = "wrong meta format"
        assert len(lines) == 2, err_info
        assert 'dnn_input_dim:' in lines[0] and 'lr_input_dim:' in lines[
            1], err_info
        res = map(int, [_.split(':')[1] for _ in lines])
        logger.info('dnn input dim: %d' % res[0])
        logger.info('lr input dim: %d' % res[1])
        return res
