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
import unittest

import paddle
from paddle.vision.models import resnet18


class TestLoadStateDictFromUrl(unittest.TestCase):
    def setUp(self):
        self.model = resnet18(pretrained=False)

    def test_load_state_dict_from_url(self):
        # Compare using load_state_dict_from_url to download and load an uncompressed weight file and a compressed weight file(ZIP)
        weight1 = paddle.hub.load_state_dict_from_url(
            url='https://paddle-hapi.bj.bcebos.com/models/resnet18.pdparams',
            model_dir="./test/test1",
            map_location="cpu",
        )
        model1 = self.model
        model1.set_state_dict(weight1)
        weight2 = paddle.hub.load_state_dict_from_url(
            url='https://x2paddle.bj.bcebos.com/resnet18.zip',
            model_dir="./test/test2",
            map_location="cpu",
        )
        model2 = self.model
        model2.set_state_dict(weight2)
        are_parameters_equal = True
        for (name1, param1), (name2, param2) in zip(
            model1.named_parameters(), model2.named_parameters()
        ):
            if name1 != name2 or not paddle.allclose(param1, param2):
                are_parameters_equal = False
                break
        assert are_parameters_equal

        # Test whether the model loads properly when the model_dir is empty and Test check_hash and map_location
        weight3 = paddle.hub.load_state_dict_from_url(
            url='https://x2paddle.bj.bcebos.com/resnet18.zip',
            model_dir="./test/test3",
            file_name="resnet18.zip",
            map_location="numpy",
        )
        model3 = self.model
        model3.set_state_dict(weight3)
        weight4 = paddle.hub.load_state_dict_from_url(
            url='https://paddle-hapi.bj.bcebos.com/models/resnet18.pdparams',
            file_name="resnet18.pdparams",
            check_hash="cf548f46534aa3560945be4b95cd11c4",
            map_location="numpy",
        )
        model4 = self.model
        model4.set_state_dict(weight4)
        are_parameters_equal = True
        for (name1, param1), (name2, param2) in zip(
            model3.named_parameters(), model4.named_parameters()
        ):
            if name1 != name2 or not paddle.allclose(param1, param2):
                are_parameters_equal = False
                break
        assert are_parameters_equal
        # Test map_location is None
        weight5 = paddle.hub.load_state_dict_from_url(
            url='https://x2paddle.bj.bcebos.com/resnet18.zip',
            model_dir="./test/test4",
        )
        model5 = self.model
        model5.set_state_dict(weight5)
        weight6 = paddle.hub.load_state_dict_from_url(
            url='https://paddle-hapi.bj.bcebos.com/models/resnet18.pdparams',
            model_dir="./test/test5",
        )
        model6 = self.model
        model6.set_state_dict(weight6)
        are_parameters_equal = True
        for (name1, param1), (name2, param2) in zip(
            model5.named_parameters(), model6.named_parameters()
        ):
            if name1 != name2 or not paddle.allclose(param1, param2):
                are_parameters_equal = False
                break
        assert are_parameters_equal


if __name__ == '__main__':
    unittest.main()
