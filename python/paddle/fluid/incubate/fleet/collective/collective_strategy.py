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

from __future__ import print_function


class DistributedStrategy(object):
    def __init__(self):
        self.use_fp16 = False
        self.use_fp32 = False
        self.local_sgd = False
        self.dgc = False
        self.hierachical_allreduce = False

    def build(self):
        if self.use_fp32 and self.use_fp16:
            self.use_fp16 = False
        if self.local_sgd and self.dgc:
            self.local_sgd
