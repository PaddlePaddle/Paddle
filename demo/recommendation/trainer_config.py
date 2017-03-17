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

from paddle.trainer_config_helpers import *

try:
    import cPickle as pickle
except ImportError:
    import pickle

is_predict = get_config_arg('is_predict', bool, False)

META_FILE = 'data/meta.bin'

with open(META_FILE, 'rb') as f:
    # load meta file
    meta = pickle.load(f)

settings(
    batch_size=1600, learning_rate=1e-3, learning_method=RMSPropOptimizer())


def construct_feature(name):
    """
    Construct movie/user features.

    This method read from meta data. Then convert feature to neural network due
    to feature type. The map relation as follow.

    * id: embedding => fc
    * embedding:
        is_sequence:  embedding => context_projection => fc => pool
        not sequence: embedding => fc
    * one_hot_dense:  fc => fc

    Then gather all features vector, and use a fc layer to combined them as
    return.

    :param name: 'movie' or 'user'
    :type name: basestring
    :return: combined feature output
    :rtype: LayerOutput
    """
    __meta__ = meta[name]['__meta__']['raw_meta']
    fusion = []
    for each_meta in __meta__:
        type_name = each_meta['type']
        slot_name = each_meta.get('name', '%s_id' % name)
        if type_name == 'id':
            slot_dim = each_meta['max']
            embedding = embedding_layer(
                input=data_layer(
                    slot_name, size=slot_dim), size=256)
            fusion.append(fc_layer(input=embedding, size=256))
        elif type_name == 'embedding':
            is_seq = each_meta['seq'] == 'sequence'
            slot_dim = len(each_meta['dict'])
            din = data_layer(slot_name, slot_dim)
            embedding = embedding_layer(input=din, size=256)
            if is_seq:
                fusion.append(
                    text_conv_pool(
                        input=embedding, context_len=5, hidden_size=256))
            else:
                fusion.append(fc_layer(input=embedding, size=256))
        elif type_name == 'one_hot_dense':
            slot_dim = len(each_meta['dict'])
            hidden = fc_layer(input=data_layer(slot_name, slot_dim), size=256)
            fusion.append(fc_layer(input=hidden, size=256))

    return fc_layer(name="%s_fusion" % name, input=fusion, size=256)


movie_feature = construct_feature("movie")
user_feature = construct_feature("user")
similarity = cos_sim(a=movie_feature, b=user_feature)
if not is_predict:
    outputs(mse_cost(input=similarity, label=data_layer('rating', size=1)))

    define_py_data_sources2(
        'data/train.list',
        'data/test.list',
        module='dataprovider',
        obj='process',
        args={'meta': meta})
else:
    outputs(similarity)
