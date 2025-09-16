#   copyright (c) 2022 paddlepaddle authors. all rights reserved.
#
# licensed under the apache license, version 2.0 (the "license");
# you may not use this file except in compliance with the license.
# you may obtain a copy of the license at
#
#     http://www.apache.org/licenses/license-2.0
#
# unless required by applicable law or agreed to in writing, software
# distributed under the license is distributed on an "as is" basis,
# without warranties or conditions of any kind, either express or implied.
# see the license for the specific language governing permissions and
# limitations under the license.

import os
import sys
import unittest

import numpy as np

sys.path.append("../../quantization")
from imperative_test_utils import (
    ImperativeLenetWithSkipQuant,
    fix_model_dict,
    train_lenet,
)

import paddle
from paddle.framework import core, set_flags
from paddle.optimizer import Adam
from paddle.quantization import ImperativeQuantAware

INFER_MODEL_SUFFIX = ".pdmodel"
INFER_PARAMS_SUFFIX = ".pdiparams"
os.environ["CPU_NUM"] = "1"
if core.is_compiled_with_cuda():
    set_flags({"FLAGS_cudnn_deterministic": True})


class TestImperativeOutSclae(unittest.TestCase):
    def test_out_scale_acc(self):
        paddle.disable_static()
        seed = 1000
        lr = 0.1

        qat = ImperativeQuantAware()

        np.random.seed(seed)
        reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=512, drop_last=True
        )

        lenet = ImperativeLenetWithSkipQuant()
        lenet = fix_model_dict(lenet)
        qat.quantize(lenet)

        adam = Adam(learning_rate=lr, parameters=lenet.parameters())
        dynamic_loss_rec = []
        lenet.train()
        loss_list = train_lenet(lenet, reader, adam)

        lenet.eval()

        path = "./save_dynamic_quant_infer_model/lenet"
        save_dir = "./save_dynamic_quant_infer_model"

        paddle.enable_static()

        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()
        exe = paddle.static.Executor(place)


if __name__ == '__main__':
    unittest.main()
