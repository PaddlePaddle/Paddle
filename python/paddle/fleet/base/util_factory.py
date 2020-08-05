#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""Fleet Utils."""
"""distributed operations"""
"""basic collective operations in python"""
"""remote file system"""

__all__ = ['UtilBase']


class UtilFactory(object):
    def _create_util(self, context):
        util = UtilBase()
        util._set_strategy(context["valid_strategy"])
        util._set_role_maker(context["role_maker"])
        return util


class UtilBase(object):
    def __init__(self):
        self.role_maker = None
        self.dist_strategy = None

    def _set_strategy(self, dist_strategy):
        self.dist_strategy = dist_strategy

    def _set_role_maker(self, role_maker):
        self.role_maker = role_maker

    '''
    def set_file_system(self, fs_client):
        self.fs_client = fs_client

    def broadcast(self):
        pass

    def all_gather(self):
        pass

    def all_reduce(self):
        pass

    def reduce_scatter(self):
        pass

    def reduce(self):
        pass

    def get_file_shard(self, files):
        pass

    def feed_gen(self, batch_size, feed_vars_dims, feeded_vars_filelist):
        pass

    def save_program(program, output_dir):
        pass

    def load_program(input_dir):
        pass

    def load_var():
        pass

    def save_var():
        pass

    def print_on_rank(self):
        pass
    '''
