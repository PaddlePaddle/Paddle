# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import os

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.distributed.flex_checkpoint.dcp.load_state_dict import (
    load_state_dict,
)


class HuggingFaceModel(nn.Layer):
    def __init__(self):
        super().__init__()
        self.huggingface = nn.Linear(2, 2, bias_attr=False)


class FCModel(nn.Layer):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 2, bias_attr=False)
        self.fc2 = nn.Linear(1, 2, bias_attr=False)


def init_hf_model_weights(model):
    with paddle.no_grad():
        w = paddle.to_tensor([[0, 1], [2, 3]], dtype="float16")
        model.huggingface.weight.set_value(w)


def save_safetensors_model(model, ckpt_path):
    import safetensors.numpy

    os.makedirs(ckpt_path, exist_ok=True)
    weight_np = model.huggingface.weight.numpy()
    file_path = os.path.join(ckpt_path, "tensor1.safetensors")
    safetensors.numpy.save_file({"huggingface.weight": weight_np}, file_path)


def test_save_load_with_missing_key_and_unexpected_keys():
    # ckpt_path = os.getenv("ckpt_path")
    ckpt_path = "/home/paddle/test/auto_parallel/hybrid_strategy/test_file"
    dist.init_parallel_env()

    hf_model = HuggingFaceModel()
    fc_model = FCModel()
    hf_model = paddle.amp.decorate(
        models=hf_model, optimizers=None, level="O2", dtype="float16"
    )
    init_hf_model_weights(hf_model)

    save_safetensors_model(hf_model, ckpt_path)

    aoa_statements = []
    aoa_config = {"aoa_statements": aoa_statements}

    try:
        load_state_dict(
            fc_model.sharded_state_dict(),
            ckpt_path,
            safetensors=True,
            aoa_config=aoa_config,
        )
        raise AssertionError
    except Exception as e:
        pass


def test_save_load_with_mapping_key_to_safetensors_file():
    ckpt_path = os.getenv("ckpt_path")
    dist.init_parallel_env()

    hf_model = HuggingFaceModel()
    fc_model = FCModel()
    hf_model = paddle.amp.decorate(
        models=hf_model, optimizers=None, level="O2", dtype="float16"
    )
    init_hf_model_weights(hf_model)

    save_safetensors_model(hf_model, ckpt_path)
    import json

    # set the key to the wrong safetensors file tensor2.safetensors, the fact file is tensor1.safetensors
    index = {
        "metadata": {"total_size": 8},
        "weight_map": {"huggingface.weight": "tensor2.safetensors"},
    }
    with open(
        os.path.join(ckpt_path, "model.safetensors.index.json"), "w"
    ) as f:
        json.dump(index, f)

    aoa_statements = [
        "huggingface.weight -> A,B ,axis = 1 \n",
        "A^T -> A  \n",
        "B^T -> B \n",
        "A -> fc1.weight ,src_dtype = 'float16', dst_dtype = 'float32' \n",
        "B -> fc2.weight ,src_dtype = 'float16', dst_dtype = 'float32' \n",
    ]
    aoa_config = {"aoa_statements": aoa_statements}

    try:
        load_state_dict(
            fc_model.sharded_state_dict(),
            ckpt_path,
            safetensors=True,
            aoa_config=aoa_config,
        )
        raise AssertionError
    except Exception as e:
        pass


def test():
    test_save_load_with_missing_key_and_unexpected_keys()
    test_save_load_with_mapping_key_to_safetensors_file()


if __name__ == "__main__":
    test()
