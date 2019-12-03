#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from . import core

__all__ = ['BoxWrapper']


class BoxWrapper(object):
    """
    box wrapper class
    """

    def __init__(self):
        self.box_wrapper = core.BoxWrapper()

    def save_model(self):
        self.box_wrapper.save_model()

    def initialize_gpu(self, conf_file, omit_var_list=None):
        if not isinstance(conf_file, str):
            raise TypeError(
                "conf_file in parameter of initialize_gpu should be str")
        if omit_var_list is None:
            omit_var_list = []
        self.box_wrapper.initialize_gpu(conf_file, omit_var_list)

    def init_metric(self,
                    name,
                    label_var,
                    pred_var,
                    is_join,
                    bucket_size=1000000):
        self.box_wrapper.init_metric(name, label_var, pred_var, is_join,
                                     bucket_size)

    def get_metric_msg(self, name):
        return self.box_wrapper.get_metric_msg(name)

    def flip_pass_flag(self):
        self.box_wrapper.flip_pass_flag()

    def finalize(self):
        self.box_wrapper.finalize()
