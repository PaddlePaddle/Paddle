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

import paddle.fluid as paddle
import paddle.v2.dataset as dataset

def conv_network():
    image = fluid.layers.data(name='image', shape=[1, 28, 28])
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    hidden = fluid.layers.batch_norm(
        fluid.layers.dropout(
            fluid.layers.simple_img_conv_pool(
                image,
                num_filters=32,
                filter_size=3,
                pool_size=3,
                pool_stride=1,
                act='relu'),
            0.1))
    prediction = fluid.layers.fc(hidden, size=10, act='softmax')
    loss = fluid.layers.cross_entropy(prediction, label)
    return loss

def main():
    fluid.train(
        reader=dataset.mnist.train(),
        num_pass=100,
        optimizer=fluid.optimizer.SGD())

if __name__ == '__main__':
    main()
