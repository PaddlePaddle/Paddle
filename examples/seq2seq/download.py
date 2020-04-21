# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
'''
Script for downloading training data.
'''
import os
import urllib
import sys

if sys.version_info >= (3, 0):
    import urllib.request
import zipfile

URLLIB = urllib
if sys.version_info >= (3, 0):
    URLLIB = urllib.request

remote_path = 'https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi'
base_path = 'data'
tar_path = os.path.join(base_path, 'en-vi')
filenames = [
    'train.en', 'train.vi', 'tst2012.en', 'tst2012.vi', 'tst2013.en',
    'tst2013.vi', 'vocab.en', 'vocab.vi'
]


def main(arguments):
    print("Downloading data......")

    if not os.path.exists(tar_path):
        if not os.path.exists(base_path):
            os.mkdir(base_path)
        os.mkdir(tar_path)

    for filename in filenames:
        url = remote_path + '/' + filename
        tar_file = os.path.join(tar_path, filename)
        URLLIB.urlretrieve(url, tar_file)
    print("Downloaded sucess......")


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
