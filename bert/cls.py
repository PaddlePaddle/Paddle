# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved. 
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
"dygraph transformer layers"

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import json
import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Linear, Layer

from hapi.text.bert.bert import BertEncoder
from hapi.model import Model


class ClsModelLayer(Model):
    """
    classify model
    """

    def __init__(self,
                 args,
                 config,
                 num_labels,
                 is_training=True,
                 return_pooled_out=True,
                 use_fp16=False):
        super(ClsModelLayer, self).__init__()
        self.config = config
        self.is_training = is_training
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
        #src_ids = data_ids[0]
        #position_ids = data_ids[1]
        #sentence_ids = data_ids[2]
        #input_mask = data_ids[3]
        #labels = data_ids[4]

        enc_output, next_sent_feat = self.bert_layer(src_ids, position_ids,
                                                     sentence_ids, input_mask)

        cls_feats = fluid.layers.dropout(
            x=next_sent_feat,
            dropout_prob=0.1,
            dropout_implementation="upscale_in_train")

        logits = self.cls_fc(cls_feats)

        return logits
        """
        logits = self.cls_fc(cls_feats)

        ce_loss, probs = fluid.layers.softmax_with_cross_entropy(
            logits=logits, label=labels, return_softmax=True)
        loss = fluid.layers.mean(x=ce_loss)

        if self.use_fp16 and self.loss_scaling > 1.0:
            loss *= self.loss_scaling

        num_seqs = fluid.layers.create_tensor(dtype='int64')
        accuracy = fluid.layers.accuracy(
            input=probs, label=labels, total=num_seqs)
        """

        return loss, accuracy
