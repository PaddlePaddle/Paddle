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

import numpy as np
from paddlenlp.transformers import (
    LlamaConfig,
    LlamaForCausalLM,
)

import paddle
import paddle.distributed as dist
from paddle.distributed.auto_parallel.high_level_api import (
    ToDistributedConfig,
    to_distributed,
)

MODEL_CLASSES = {
    "llama": (LlamaConfig, LlamaForCausalLM),
}

BATCH_SIZE = 4
BATCH_NUM = 3
SEQ_LENGTH = 1024


class RandomDataset(paddle.io.Dataset):
    def __init__(self, images, labels, num_samples):
        self.images = images
        self.labels = labels
        self.num_samples = num_samples

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def __len__(self):
        return self.num_samples


def llama_init():
    config_class, model_class = MODEL_CLASSES["llama"]
    config = config_class()
    config.hidden_size = 2048
    config.intermediate_size = 4096
    config.num_hidden_layers = 2
    config.seq_length = 1024
    config.max_position_embeddings = 1024
    config.alibi = False
    config.sequence_parallel = False
    config.fuse_attention_qkv = False

    # with paddle.LazyGuard():
    #     model = model_class.from_config(config, dtype="float32")
    model = model_class.from_config(config, dtype="float32")

    return config, model


# create mesh
mesh = dist.ProcessMesh([0, 1, 2, 3, 4, 5, 6, 7], dim_names=["x"])

model_config, model = llama_init()
print(f"llama config is {model_config}")
print(f"llama model is {model}")
input_seqs = np.random.randint(
    low=0, high=1024, size=(BATCH_SIZE * BATCH_NUM, SEQ_LENGTH)
).astype("int64")
labels = np.random.randint(
    low=0, high=1024, size=(BATCH_SIZE * BATCH_NUM, SEQ_LENGTH)
).astype("int64")
dataset = RandomDataset(input_seqs, labels, BATCH_SIZE * BATCH_NUM)
loader = paddle.io.DataLoader(dataset, batch_size=BATCH_SIZE)
opt = paddle.optimizer.SGD(learning_rate=0.1, parameters=model.parameters())

# # shard dataloader
dist_loader = dist.shard_dataloader(loader, meshes=[mesh])
# # config: input_spec
input_seq_spec = paddle.static.InputSpec(
    [BATCH_SIZE, SEQ_LENGTH], 'float32', 'input_seq', True
)
dist_config = ToDistributedConfig()
dist_config.input_spec = [input_seq_spec]
dist_config.num_hidden_layers = model_config.num_hidden_layers
# # wrap model by using **to_distributed**
dist_model = to_distributed(model, mesh, dist_config)
# dist_model = model

dist_model.train()
for batch_id, (input_seq, label) in enumerate(dist_loader()):
    # dynamic
    print(f"input_seq is {input_seq}, labels is {label}")
    (loss, logits) = dist_model(input_ids=input_seq, labels=label)
    print(f"batch: {batch_id}, loss is {loss}")
    loss.backward()
    opt.step()
    opt.clear_grad()
    # if batch_id == 2:
    #     breakpoint()
