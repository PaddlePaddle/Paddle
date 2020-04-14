#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""BERT fine-tuning in Paddle Dygraph Mode."""

import paddle.fluid as fluid
from hapi.metrics import Accuracy
from hapi.configure import Config
from hapi.model import set_device, Model, SoftmaxWithCrossEntropy, Input

from cls import ClsModelLayer
import hapi.text.tokenizer.tokenization as tokenization
from hapi.text.bert import Optimizer, BertConfig, BertDataLoader, BertInputExample


def train():

    config = Config(yaml_file="./bert.yaml")
    config.build()
    config.Print()

    device = set_device("gpu" if config.use_cuda else "cpu")
    fluid.enable_dygraph(device)

    bert_config = BertConfig(config.bert_config_path)
    bert_config.print_config()

    trainer_count = fluid.dygraph.parallel.Env().nranks

    tokenizer = tokenization.FullTokenizer(
        vocab_file=config.vocab_path, do_lower_case=config.do_lower_case)

    def mnli_line_processor(line_id, line):
        if line_id == "0":
            return None
        uid = tokenization.convert_to_unicode(line[0])
        text_a = tokenization.convert_to_unicode(line[8])
        text_b = tokenization.convert_to_unicode(line[9])
        label = tokenization.convert_to_unicode(line[-1])
        if label not in ["contradiction", "entailment", "neutral"]:
            label = "contradiction"
        return BertInputExample(
            uid=uid, text_a=text_a, text_b=text_b, label=label)

    bert_dataloader = BertDataLoader(
        "./data/glue_data/MNLI/train.tsv",
        tokenizer, ["contradiction", "entailment", "neutral"],
        max_seq_length=64,
        batch_size=32,
        line_processor=mnli_line_processor)

    num_train_examples = len(bert_dataloader.dataset)
    max_train_steps = config.epoch * num_train_examples // config.batch_size // trainer_count
    warmup_steps = int(max_train_steps * config.warmup_proportion)

    print("Trainer count: %d" % trainer_count)
    print("Num train examples: %d" % num_train_examples)
    print("Max train steps: %d" % max_train_steps)
    print("Num warmup steps: %d" % warmup_steps)

    inputs = [
        Input(
            [None, None], 'int64', name='src_ids'), Input(
                [None, None], 'int64', name='pos_ids'), Input(
                    [None, None], 'int64', name='sent_ids'), Input(
                        [None, None], 'float32', name='input_mask')
    ]

    labels = [Input([None, 1], 'int64', name='label')]

    cls_model = ClsModelLayer(
        config,
        bert_config,
        len(["contradiction", "entailment", "neutral"]),
        is_training=True,
        return_pooled_out=True)

    optimizer = Optimizer(
        warmup_steps=warmup_steps,
        num_train_steps=max_train_steps,
        learning_rate=config.learning_rate,
        model_cls=cls_model,
        weight_decay=config.weight_decay,
        scheduler=config.lr_scheduler,
        loss_scaling=config.loss_scaling,
        parameter_list=cls_model.parameters())

    cls_model.prepare(
        optimizer,
        SoftmaxWithCrossEntropy(),
        Accuracy(topk=(1, 2)),
        inputs,
        labels,
        device=device)

    cls_model.bert_layer.init_parameters(
        config.init_pretraining_params, verbose=config.verbose)

    cls_model.fit(train_data=bert_dataloader.dataloader, epochs=config.epoch)

    return cls_model


if __name__ == '__main__':
    cls_model = train()
