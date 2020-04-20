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
from hapi.text.bert import BertEncoder
from paddle.fluid.dygraph import Linear, Layer
from hapi.model import set_device, Model, SoftmaxWithCrossEntropy, Input
import hapi.text.tokenizer.tokenization as tokenization
from hapi.text.bert import Optimizer, BertConfig, BertDataLoader, BertInputExample


class ClsModelLayer(Model):
    """
    classify model
    """

    def __init__(self,
                 args,
                 config,
                 num_labels,
                 return_pooled_out=True,
                 use_fp16=False):
        super(ClsModelLayer, self).__init__()
        self.config = config
        self.use_fp16 = use_fp16
        self.loss_scaling = args.loss_scaling

        self.bert_layer = BertEncoder(
            config=self.config, return_pooled_out=True, use_fp16=self.use_fp16)

        self.cls_fc = Linear(
            input_dim=self.config["hidden_size"],
            output_dim=num_labels,
            param_attr=fluid.ParamAttr(
                name="cls_out_w",
                initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
            bias_attr=fluid.ParamAttr(
                name="cls_out_b", initializer=fluid.initializer.Constant(0.)))

    def forward(self, src_ids, position_ids, sentence_ids, input_mask):
        """
        forward
        """

        enc_output, next_sent_feat = self.bert_layer(src_ids, position_ids,
                                                     sentence_ids, input_mask)

        cls_feats = fluid.layers.dropout(
            x=next_sent_feat,
            dropout_prob=0.1,
            dropout_implementation="upscale_in_train")

        pred = self.cls_fc(cls_feats)

        return pred


def main():

    config = Config(yaml_file="./bert.yaml")
    config.build()
    config.Print()

    device = set_device("gpu" if config.use_cuda else "cpu")
    fluid.enable_dygraph(device)

    bert_config = BertConfig(config.bert_config_path)
    bert_config.print_config()

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

    train_dataloader = BertDataLoader(
        "./data/glue_data/MNLI/train.tsv",
        tokenizer, ["contradiction", "entailment", "neutral"],
        max_seq_length=config.max_seq_len,
        batch_size=config.batch_size,
        line_processor=mnli_line_processor,
        mode="leveldb",
        phase="train")

    dev_dataloader = BertDataLoader(
        "./data/glue_data/MNLI/dev_matched.tsv",
        tokenizer, ["contradiction", "entailment", "neutral"],
        max_seq_length=config.max_seq_len,
        batch_size=config.batch_size,
        line_processor=mnli_line_processor,
        shuffle=False,
        phase="predict")

    trainer_count = fluid.dygraph.parallel.Env().nranks
    num_train_examples = len(train_dataloader.dataset)
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

    # do train
    cls_model.fit(train_data=train_dataloader.dataloader,
                  epochs=config.epoch,
                  save_dir=config.checkpoints)

    # do eval
    cls_model.evaluate(
        eval_data=test_dataloader.dataloader, batch_size=config.batch_size)


if __name__ == '__main__':
    main()
