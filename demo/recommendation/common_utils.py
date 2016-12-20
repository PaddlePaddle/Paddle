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
from paddle.trainer.PyDataProvider2 import *


def meta_to_header(meta, name):
    metas = meta[name]['__meta__']['raw_meta']
    for each_meta in metas:
        slot_name = each_meta.get('name', '%s_id' % name)
        if each_meta['type'] == 'id':
            yield slot_name, integer_value(each_meta['max'])
        elif each_meta['type'] == 'embedding':
            is_seq = each_meta['seq'] == 'sequence'
            yield slot_name, integer_value(
                len(each_meta['dict']),
                seq_type=SequenceType.SEQUENCE
                if is_seq else SequenceType.NO_SEQUENCE)
        elif each_meta['type'] == 'one_hot_dense':
            yield slot_name, dense_vector(len(each_meta['dict']))
