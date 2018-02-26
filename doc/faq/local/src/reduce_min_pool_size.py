#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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


@provider(min_pool_size=0, ...)
def process(settings, filename):
    os.system('shuf %s > %s.shuf' % (filename, filename))  # shuffle before.
    with open('%s.shuf' % filename, 'r') as f:
        for line in f:
            yield get_sample_from_line(line)
