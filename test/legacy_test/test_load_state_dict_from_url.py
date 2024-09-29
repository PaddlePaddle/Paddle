# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle.vision.models import resnet18


def test_load_state_dict_from_url():
    # Verify the consistency of the local files loaded by paddle.load with the local files loaded load_state_dict_from_url

    weight_path = '/paddle/test_zty/test/resnet18.pdparams'
    model1 = resnet18(pretrained=False)
    model1.set_state_dict(paddle.load(weight_path))
    weight = paddle.hapi.hub.load_state_dict_from_url(
        url='https://paddle-hapi.bj.bcebos.com/models/resnet18.pdparams',
        model_dir="/paddle/test_zty/test",
    )
    model2 = resnet18(pretrained=False)
    model2.set_state_dict(weight)
    are_parameters_equal = True
    for (name1, param1), (name2, param2) in zip(
        model1.named_parameters(), model2.named_parameters()
    ):
        if name1 != name2 or not paddle.allclose(param1, param2):
            are_parameters_equal = False
            break
    print(
        "Whether the model weight files loaded by the two methods are consistent:",
        are_parameters_equal,
    )

    # Verify the consistency of the local file loaded by paddle.load with the model weight file that load_state_dict_from_url downloaded and loaded

    weight_path = '/paddle/test_zty/test/resnet18.pdparams'
    model1 = resnet18(pretrained=False)
    model1.set_state_dict(paddle.load(weight_path))
    weight = paddle.hapi.hub.load_state_dict_from_url(
        url='https://paddle-hapi.bj.bcebos.com/models/resnet18.pdparams',
        model_dir="/paddle/test_zty/test1",
    )
    model2 = resnet18(pretrained=False)
    model2.set_state_dict(weight)
    are_parameters_equal = True
    for (name1, param1), (name2, param2) in zip(
        model1.named_parameters(), model2.named_parameters()
    ):
        if name1 != name2 or not paddle.allclose(param1, param2):
            are_parameters_equal = False
            break
    print(
        "Whether the model weight files loaded by the two methods are consistent:",
        are_parameters_equal,
    )

    #  Verify the consistency of the local file loaded by paddle.load with the model weight file that load_state_dict_from_url downloaded, unzipped (ZIP) and loaded
    weight_path = '/paddle/test_zty/test/resnet18.pdparams'
    model1 = resnet18(pretrained=False)
    model1.set_state_dict(paddle.load(weight_path))
    weight = paddle.hapi.hub.load_state_dict_from_url(
        url='http://127.0.0.1:9100/download/resnet18.zip',
        model_dir="/paddle/test_zty/test2",
    )
    model2 = resnet18(pretrained=False)
    model2.set_state_dict(weight)
    are_parameters_equal = True
    for (name1, param1), (name2, param2) in zip(
        model1.named_parameters(), model2.named_parameters()
    ):
        if name1 != name2 or not paddle.allclose(param1, param2):
            are_parameters_equal = False
            break
    print(
        "Whether the model weight files loaded by the two methods are consistent:",
        are_parameters_equal,
    )


if __name__ == '__main__':
    test_load_state_dict_from_url()
