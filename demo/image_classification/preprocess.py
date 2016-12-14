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

from paddle.utils.preprocess_img import ImageClassificationDatasetCreater
from optparse import OptionParser


def option_parser():
    parser = OptionParser(usage="usage: python preprcoess.py "\
                          "-i data_dir [options]")
    parser.add_option(
        "-i",
        "--input",
        action="store",
        dest="input",
        help="Input data directory.")
    parser.add_option(
        "-s",
        "--size",
        action="store",
        dest="size",
        help="Processed image size.")
    parser.add_option(
        "-c",
        "--color",
        action="store",
        dest="color",
        help="whether to use color images.")
    return parser.parse_args()


if __name__ == '__main__':
    options, args = option_parser()
    data_dir = options.input
    processed_image_size = int(options.size)
    color = options.color == "1"
    data_creator = ImageClassificationDatasetCreater(
        data_dir, processed_image_size, color)
    data_creator.train_list_name = "train.txt"
    data_creator.test_list_name = "test.txt"
    data_creator.num_per_batch = 1000
    data_creator.overwrite = True
    data_creator.create_batches()
