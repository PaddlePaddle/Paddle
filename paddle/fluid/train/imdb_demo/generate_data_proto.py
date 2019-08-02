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

import paddle.fluid as fluid

dataset = fluid.DatasetFactory().create_dataset()
dataset.set_batch_size(128)
dataset.set_pipe_command("python imdb_reader.py")

data = fluid.layers.data(name="words", shape=[1], dtype="int64", lod_level=1)
label = fluid.layers.data(name="label", shape=[1], dtype="int64")

dataset.set_use_var([data, label])
desc = dataset.proto_desc

with open("data.proto", "w") as f:
    f.write(dataset.desc())
