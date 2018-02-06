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

logging.basicConfig(
    format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s')
logging.getLogger().setLevel(logging.INFO)


def load_feature_c(file):
    """
    Load feature extracted by C++ interface.
    Return a list.
    file: feature file.
    """
    features = []
    f = open(file, 'r')
    for line in f:
        sample = []
        for slot in line.strip().split(";"):
            fea = [float(val) for val in slot.strip().split()]
            if fea:
                sample.append(fea)
        features.append(sample)
    f.close()
    return features


def load_feature_py(feature_dir):
    """
    Load feature extracted by python interface.
    Return a dictionary.
    feature_dir: directory of feature file.
    """
    file_list = os.listdir(feature_dir)
    file_list = [os.path.join(feature_dir, f) for f in file_list]
    features = {}
    for file_name in file_list:
        with open(file_name, 'rb') as f:
            feature = cPickle.load(f)
            features.update(feature)
            logging.info('Load feature file %s', file_name)
    return features


if __name__ == '__main__':
    print load_feature_py(sys.argv[1])
    #print load_feature_c(sys.argv[1]) 
