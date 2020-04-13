#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import os
import io
import time
import argparse
import paddle.fluid as fluid

from paddle.fluid.dygraph import to_variable
from hapi.text.text import PrePostProcessLayer
from hapi.text.bert.bert import BertConfig
from cls import ClsModelLayer
from hapi.text.bert.optimization import Optimizer
from hapi.text.bert.utils.args import ArgumentGroup, print_arguments, check_cuda
from hapi.model import set_device, Model, SoftmaxWithCrossEntropy, Input
from hapi.metrics import Accuracy

from hapi.text.bert.dataloader import SingleSentenceDataLoader, BertInputExample
import hapi.text.tokenizer.tokenization as tokenization

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
model_g = ArgumentGroup(parser, "model", "model configuration and paths.")
model_g.add_arg("bert_config_path",      str,  "./config/bert_config.json",  "Path to the json file for bert model config.")
model_g.add_arg("init_checkpoint",       str,  None,                         "Init checkpoint to resume training from.")
model_g.add_arg("init_pretraining_params",  str,  None,
                "Init pre-training params which preforms fine-tuning from. If the "
                "arg 'init_checkpoint' has been set, this argument wouldn't be valid.")
model_g.add_arg("checkpoints",           str,  "checkpoints",                "Path to save checkpoints.")

train_g = ArgumentGroup(parser, "training", "training options.")
train_g.add_arg("epoch",             int,    100,     "Number of epoches for training.")
train_g.add_arg("learning_rate",     float,  0.0001,  "Learning rate used to train with warmup.")
train_g.add_arg("lr_scheduler",      str,    "linear_warmup_decay",
                "scheduler of learning rate.", choices=['linear_warmup_decay', 'noam_decay'])
train_g.add_arg("weight_decay",      float,  0.01,    "Weight decay rate for L2 regularizer.")
train_g.add_arg("warmup_proportion",     float,  0.1,                         "Proportion of training steps to perform linear learning rate warmup for.")
train_g.add_arg("save_steps",        int,    10000,   "The steps interval to save checkpoints.")
train_g.add_arg("validation_steps",  int,    1000,    "The steps interval to evaluate model performance.")
train_g.add_arg("loss_scaling",      float,  1.0,
                "Loss scaling factor for mixed precision training, only valid when use_fp16 is enabled.")

log_g = ArgumentGroup(parser,     "logging", "logging related.")
log_g.add_arg("skip_steps",          int,    10,    "The steps interval to print loss.")

data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options")
data_g.add_arg("data_dir",            str,  None,       "Path to training data.")
data_g.add_arg("vocab_path",          str,  None,       "Vocabulary path.")
data_g.add_arg("max_seq_len",         int,  512,                   "Tokens' number of the longest seqence allowed.")
data_g.add_arg("batch_size",          int,  32,
               "The total number of examples in one batch for training, see also --in_tokens.")
data_g.add_arg("in_tokens",           bool, False,
               "If set, the batch size will be the maximum number of tokens in one batch. "
               "Otherwise, it will be the maximum number of examples in one batch.")
data_g.add_arg("do_lower_case", bool, True,
               "Whether to lower case the input text. Should be True for uncased models and False for cased models.")
data_g.add_arg("random_seed",   int,  5512,     "Random seed.")

run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
run_type_g.add_arg("use_cuda",                     bool,   True,   "If set, use GPU for training.")
run_type_g.add_arg("shuffle",                      bool,   True,  "")
run_type_g.add_arg("task_name",                    str,    None,
                   "The name of task to perform fine-tuning, should be in {'xnli', 'mnli', 'cola', 'mrpc'}.")
run_type_g.add_arg("do_train",                     bool,   True,  "Whether to perform training.")
run_type_g.add_arg("do_test",                      bool,   False,  "Whether to perform evaluation on test data set.")
run_type_g.add_arg("use_data_parallel", bool, False,  "The flag indicating whether to shuffle instances in each pass.")
run_type_g.add_arg("enable_ce", bool, False,  help="The flag indicating whether to run the task for continuous evaluation.")

args = parser.parse_args()

def create_data(batch):
    """
    convert data to variable
    """
    src_ids = to_variable(batch[0], "src_ids")
    position_ids = to_variable(batch[1], "position_ids")
    sentence_ids = to_variable(batch[2], "sentence_ids")
    input_mask = to_variable(batch[3], "input_mask")
    labels = to_variable(batch[4], "labels")
    labels.stop_gradient = True
    return src_ids, position_ids, sentence_ids, input_mask, labels

def train(args):

    device = set_device("gpu" if args.use_cuda else "cpu")
    fluid.enable_dygraph(device)

    bert_config = BertConfig(args.bert_config_path)
    bert_config.print_config()

    if not (args.do_train or args.do_test):
        raise ValueError("For args `do_train`, `do_test`, at "
                        "least one of them must be True.")

    trainer_count = fluid.dygraph.parallel.Env().nranks

    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_path, do_lower_case=args.do_lower_case)

    def mnli_line_processor(line_id, line):
        if line_id == "0":
            return None
        uid = tokenization.convert_to_unicode(line[0])
        text_a = tokenization.convert_to_unicode(line[8])
        text_b = tokenization.convert_to_unicode(line[9])
        label = tokenization.convert_to_unicode(line[-1])
        if label not in ["contradiction", "entailment", "neutral"]:
            label = "contradiction"
        return BertInputExample(uid=uid, text_a=text_a, text_b=text_b, label=label)

    bert_dataloader = SingleSentenceDataLoader("./data/glue_data/MNLI/train.tsv", tokenizer, ["contradiction", "entailment", "neutral"],
        max_seq_length=64, batch_size=32, line_processor=mnli_line_processor)

    num_train_examples = len(bert_dataloader.dataset)
    max_train_steps = args.epoch * num_train_examples // args.batch_size // trainer_count
    warmup_steps = int(max_train_steps * args.warmup_proportion)

    print("Trainer count: %d" % trainer_count)
    print("Num train examples: %d" % num_train_examples)
    print("Max train steps: %d" % max_train_steps)
    print("Num warmup steps: %d" % warmup_steps)

    if args.use_data_parallel:
        strategy = fluid.dygraph.parallel.prepare_context()

    inputs = [Input([None, None], 'int64', name='src_ids'),
              Input([None, None], 'int64', name='pos_ids'),
              Input([None, None], 'int64', name='sent_ids'),
              Input([None, None], 'float32', name='input_mask')]

    labels = [Input([None, 1], 'int64', name='label')]

    cls_model = ClsModelLayer(
                            args,
                            bert_config,
                            3,
                            is_training=True,
                            return_pooled_out=True)

    optimizer = Optimizer(
                    warmup_steps=warmup_steps,
                    num_train_steps=max_train_steps,
                    learning_rate=args.learning_rate,
                    model_cls=cls_model,
                    weight_decay=args.weight_decay,
                    scheduler=args.lr_scheduler,
                    loss_scaling=args.loss_scaling,
                    parameter_list=cls_model.parameters())

    cls_model.prepare(
        optimizer,
        SoftmaxWithCrossEntropy(),
        Accuracy(topk=(1, 2)),
        inputs,
        labels,
        device=device)

    cls_model.bert_layer.init_parameters(args.init_pretraining_params, verbose=True)

    cls_model.fit(train_data=bert_dataloader.dataloader, epochs=args.epoch)

    return cls_model


if __name__ == '__main__':

    print_arguments(args)
    check_cuda(args.use_cuda)

    if args.do_train:
        cls_model = train(args)
