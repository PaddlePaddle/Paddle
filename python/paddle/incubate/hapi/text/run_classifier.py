# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import collections
from functools import partial

from paddle.io import DataLoader
from glue import *
from data_utils import *
from model_utils import *
from bert import *


def convert_examples_to_features(example,
                                 tokenizer=None,
                                 truncate_length=512,
                                 cls_token=None,
                                 sep_token=None,
                                 class_labels=None,
                                 label_alias=None,
                                 vocab=None,
                                 is_test=False):
    """convert glue examples into necessary features"""
    if not is_test:
        label_dtype = 'int32' if class_labels else 'float32'
        # get the label
        label = example[-1]
        example = example[:-1]
        #create label maps if classification task
        if class_labels:
            label_map = {}
            for (i, l) in enumerate(class_labels):
                label_map[l] = i
            if label_alias:
                for key in label_alias:
                    label_map[key] = label_map[label_alias[key]]
            label = label_map[label]
        label = np.array([label], dtype=label_dtype)

    # tokenize raw text
    tokens_raw = [tokenizer(l) for l in example]
    # truncate to the truncate_length,
    tokens_trun = truncate_seqs_equal(tokens_raw, truncate_length)
    # concate the sequences with special tokens
    tokens_trun[0] = [cls_token] + tokens_trun[0]
    tokens, segment_ids, _ = concat_sequences(tokens_trun,
                                              [[sep_token]] * len(tokens_trun))
    # convert the token to ids
    input_ids = vocab[tokens]
    valid_length = len(input_ids)
    if not is_test:
        return input_ids, segment_ids, valid_length, label
    else:
        return input_ids, segment_ids, valid_length


TASK_CLASSES = {"mnli": (GlueMNLI, ), }

MODEL_CLASSES = {"bert": (BertForSequenceClassification, BertTokenizer), }


def do_train(args):
    device = set_device("gpu" if args.use_cuda else "cpu")
    fluid.enable_dygraph(device) if args.eager_run else None

    args.task_name = args.task_name.lower()
    dataset_class = TASK_CLASSES[args.task_name]
    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    train_dataset = dataset_class("train")
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    trans_func = partial(convert_examples_to_features, tokenizer)
    train_dataset = WrapDataset(train_dataset).apply(trans_func)
    train_batch_sampler = SamplerHelper(train_dataset).shuffle().batch(
        batch_size=args.batch_size).shard()
    batchify_fn = Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token),  # input
        Pad(axis=0, pad_val=0),  # segment
        Stack(),  # length
        Stack(label_dtype))  # label
    data_loader = DataLoader(
        dataset=train_dataset,
        batch_sampler=train_batch_sampler,
        places=device,
        collate_fn=batchify_fn,
        num_workers=0,
        return_list=True)

    model = model_class.from_pretrained(
        args.model_name_or_path, num_labels=len(train_dataset.get_labels()))


if __name__ == "__main__":
    do_train(args)
