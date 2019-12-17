#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import numpy as np
import copy
import pickle
import os
from functools import partial
import logging
import time
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import argparse
import random
import sys
import math

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)

batch_size = 100
ncards = 4
nreaders = 4
nscopes = 30
learning_rate = 0.1
is_profile = False
sync_steps = 1


def parse_args():
    parser = argparse.ArgumentParser("gnn")
    parser.add_argument(
        '--train_path',
        type=str,
        default='./data/diginetica/train.txt',
        help='dir of training data')
    parser.add_argument(
        '--config_path',
        type=str,
        default='./data/diginetica/config.txt',
        help='dir of config')
    parser.add_argument(
        '--model_path',
        type=str,
        default='./saved_model',
        help="path of model parameters")
    parser.add_argument(
        '--epoch_num',
        type=int,
        default=30,
        help='number of epochs to train for')
    parser.add_argument(
        '--batch_size', type=int, default=100, help='input batch size')
    parser.add_argument(
        '--hidden_size', type=int, default=100, help='hidden state size')
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument(
        '--emb_lr_rate', type=float, default=0.5, help='learning rate')
    parser.add_argument(
        '--step', type=int, default=1, help='gnn propogation steps')
    parser.add_argument(
        '--lr_dc', type=float, default=0.1, help='learning rate decay rate')
    parser.add_argument(
        '--lr_dc_step',
        type=int,
        default=3,
        help='the number of steps after which the learning rate decay')
    parser.add_argument(
        '--use_cuda', type=int, default=0, help='whether to use gpu')
    parser.add_argument(
        '--use_parallel',
        type=int,
        default=1,
        help='whether to use parallel executor')
    return parser.parse_args()


def network(batch_size, items_num, hidden_size, step, rate):
    stdv = 1.0 / math.sqrt(hidden_size)

    items = layers.data(
        name="items",
        shape=[batch_size, -1, 1],
        dtype="int64",
        append_batch_size=False)  #[bs, uniq_max, 1]
    seq_index = layers.data(
        name="seq_index",
        shape=[batch_size, -1],
        dtype="int64",
        append_batch_size=False)  #[-1(seq_max)*batch_size, 1]
    last_index = layers.data(
        name="last_index",
        shape=[batch_size],
        dtype="int64",
        append_batch_size=False)  #[batch_size, 1]
    adj_in = layers.data(
        name="adj_in",
        shape=[batch_size, -1, -1],
        dtype="float32",
        append_batch_size=False)
    adj_out = layers.data(
        name="adj_out",
        shape=[batch_size, -1, -1],
        dtype="float32",
        append_batch_size=False)
    mask = layers.data(
        name="mask",
        shape=[batch_size, -1, 1],
        dtype="float32",
        append_batch_size=False)
    label = layers.data(
        name="label",
        shape=[batch_size, 1],
        dtype="int64",
        append_batch_size=False)

    items_emb = layers.embedding(
        input=items,
        is_sparse=True,
        param_attr=fluid.ParamAttr(
            name="emb",
            learning_rate=rate,
            initializer=fluid.initializer.Uniform(
                low=-stdv, high=stdv)),
        size=[items_num, hidden_size])  #[batch_size, uniq_max, h]
    data_feed = [items, seq_index, last_index, adj_in, adj_out, mask, label]

    pre_state = items_emb
    for i in range(step):
        pre_state = layers.reshape(
            x=pre_state, shape=[batch_size, -1, hidden_size])
        state_in = layers.fc(
            input=pre_state,
            name="state_in",
            size=hidden_size,
            act=None,
            num_flatten_dims=2,
            param_attr=fluid.ParamAttr(initializer=fluid.initializer.Uniform(
                low=-stdv, high=stdv)),
            bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Uniform(
                low=-stdv, high=stdv)))  #[batch_size, uniq_max, h]
        state_out = layers.fc(
            input=pre_state,
            name="state_out",
            size=hidden_size,
            act=None,
            num_flatten_dims=2,
            param_attr=fluid.ParamAttr(initializer=fluid.initializer.Uniform(
                low=-stdv, high=stdv)),
            bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Uniform(
                low=-stdv, high=stdv)))  #[batch_size, uniq_max, h]

        state_adj_in = layers.matmul(adj_in,
                                     state_in)  #[batch_size, uniq_max, h]
        state_adj_out = layers.matmul(adj_out,
                                      state_out)  #[batch_size, uniq_max, h]

        gru_input = layers.concat([state_adj_in, state_adj_out], axis=2)

        gru_input = layers.reshape(x=gru_input, shape=[-1, hidden_size * 2])
        gru_fc = layers.fc(input=gru_input,
                           name="gru_fc",
                           size=3 * hidden_size,
                           bias_attr=False)
        pre_state, _, _ = fluid.layers.gru_unit(
            input=gru_fc,
            hidden=layers.reshape(
                x=pre_state, shape=[-1, hidden_size]),
            size=3 * hidden_size)

    final_state = pre_state
    seq_index = layers.reshape(seq_index, shape=[-1])
    seq = layers.gather(final_state, seq_index)  #[batch_size*-1(seq_max), h]
    last = layers.gather(final_state, last_index)  #[batch_size, h]

    seq = layers.reshape(
        seq, shape=[batch_size, -1, hidden_size])  #[batch_size, -1(seq_max), h]
    last = layers.reshape(
        last, shape=[batch_size, hidden_size])  #[batch_size, h]

    seq_fc = layers.fc(
        input=seq,
        name="seq_fc",
        size=hidden_size,
        bias_attr=False,
        act=None,
        num_flatten_dims=2,
        param_attr=fluid.ParamAttr(initializer=fluid.initializer.Uniform(
            low=-stdv, high=stdv)))  #[batch_size, -1(seq_max), h]
    last_fc = layers.fc(input=last,
                        name="last_fc",
                        size=hidden_size,
                        bias_attr=False,
                        act=None,
                        num_flatten_dims=1,
                        param_attr=fluid.ParamAttr(
                            initializer=fluid.initializer.Uniform(
                                low=-stdv, high=stdv)))  #[bathc_size, h]

    seq_fc_t = layers.transpose(
        seq_fc, perm=[1, 0, 2])  #[-1(seq_max), batch_size, h]
    add = layers.elementwise_add(seq_fc_t,
                                 last_fc)  #[-1(seq_max), batch_size, h]
    b = layers.create_parameter(
        shape=[hidden_size],
        dtype='float32',
        default_initializer=fluid.initializer.Constant(value=0.0))  #[h]
    add = layers.elementwise_add(add, b)  #[-1(seq_max), batch_size, h]

    add_sigmoid = layers.sigmoid(add)  #[-1(seq_max), batch_size, h] 
    add_sigmoid = layers.transpose(
        add_sigmoid, perm=[1, 0, 2])  #[batch_size, -1(seq_max), h]

    weight = layers.fc(input=add_sigmoid,
                       name="weight_fc",
                       size=1,
                       act=None,
                       num_flatten_dims=2,
                       bias_attr=False,
                       param_attr=fluid.ParamAttr(
                           initializer=fluid.initializer.Uniform(
                               low=-stdv, high=stdv)))  #[batch_size, -1, 1]
    weight *= mask
    weight_mask = layers.elementwise_mul(seq, weight, axis=0)
    global_attention = layers.reduce_sum(weight_mask, dim=1)

    final_attention = layers.concat(
        [global_attention, last], axis=1)  #[batch_size, 2*h]
    final_attention_fc = layers.fc(
        input=final_attention,
        name="fina_attention_fc",
        size=hidden_size,
        bias_attr=False,
        act=None,
        param_attr=fluid.ParamAttr(initializer=fluid.initializer.Uniform(
            low=-stdv, high=stdv)))  #[batch_size, h]

    all_vocab = layers.create_global_var(
        shape=[items_num - 1, 1],
        value=0,
        dtype="int64",
        persistable=True,
        name="all_vocab")

    all_emb = layers.embedding(
        input=all_vocab,
        is_sparse=True,
        param_attr=fluid.ParamAttr(
            name="emb",
            learning_rate=rate,
            initializer=fluid.initializer.Uniform(
                low=-stdv, high=stdv)),
        size=[items_num, hidden_size])  #[all_vocab, h]

    logits = layers.matmul(
        x=final_attention_fc, y=all_emb,
        transpose_y=True)  #[batch_size, all_vocab]
    softmax = layers.softmax_with_cross_entropy(
        logits=logits, label=label)  #[batch_size, 1]
    loss = layers.reduce_mean(softmax)  # [1]
    #fluid.layers.Print(loss)
    acc = layers.accuracy(input=logits, label=label, k=20)
    return loss, acc, data_feed, [items_emb, all_emb]


def train():
    args = parse_args()
    lr = args.lr
    rate = args.emb_lr_rate
    train_data_dir = "./gnn_data_new_8"
    filelist = [
        os.path.join(train_data_dir, f) for f in os.listdir(train_data_dir)
        if os.path.isfile(os.path.join(train_data_dir, f))
    ][:]

    items_num = read_config(args.config_path)
    loss, acc, data_vars, cut_list = network(batch_size, items_num,
                                             args.hidden_size, args.step, rate)

    print("card: %d, thread: %d, lr: %f, lr_rate: %f, scope: %d, sync_step: %d"
          % (ncards, nreaders, lr, rate, nscopes, sync_steps))

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    step_per_epoch = 750000 // batch_size
    """
    opt = fluid.optimizer.SGD(
	learning_rate=fluid.layers.exponential_decay(
	    learning_rate=args.lr,
	    decay_steps=step_per_epoch * 10,
	    decay_rate=args.lr_dc),
	regularization=fluid.regularizer.L2DecayRegularizer(regularization_coeff=args.l2))
    """
    opt = fluid.optimizer.SGD(lr)
    opt = fluid.optimizer.PipelineOptimizer(
        opt,
        cut_list=[cut_list, [loss, acc]],
        place_list=[fluid.CPUPlace(), fluid.CUDAPlace(0), fluid.CPUPlace()],
        concurrency_list=[1, 1, nreaders],
        queue_size=nscopes,
        sync_steps=sync_steps)
    opt.minimize(loss)

    exe.run(fluid.default_startup_program())

    all_vocab = fluid.global_scope().var("all_vocab").get_tensor()
    all_vocab.set(
        np.arange(1, items_num).astype("int64").reshape((-1, 1)), place)

    logger.info("begin train")

    dataset = fluid.DatasetFactory().create_dataset("FileInstantDataset")
    dataset.set_use_var(data_vars)
    dataset.set_batch_size(batch_size)
    dataset.set_filelist(filelist)

    total_time = []
    start_time = time.time()
    loss_sum = 0.0
    acc_sum = 0.0
    global_step = 0

    for i in range(25):
        logger.info("begin epoch %d" % (i))
        epoch_sum = []
        random.shuffle(filelist)
        dataset.set_filelist(filelist)
        exe.train_from_dataset(
            fluid.default_main_program(),
            dataset,
            thread=ncards,
            debug=is_profile,
            fetch_list=[loss, acc],
            fetch_info=["loss", "acc"],
            print_period=1)
        model_path = args.model_path
        model_path += "_" + str(lr) + "_" + str(rate)
        save_dir = model_path + "/epoch_" + str(i)
        fetch_vars = [loss, acc]
        feed_list = [
            "items", "seq_index", "last_index", "adj_in", "adj_out", "mask",
            "label"
        ]
        fluid.io.save_inference_model(save_dir, feed_list, fetch_vars, exe)


class Data():
    def __init__(self, path, shuffle=False):
        data = pickle.load(open(path, 'rb'))
        self.shuffle = shuffle
        self.length = len(data[0])
        self.input = list(zip(data[0], data[1]))

    def make_data(self, cur_batch, batch_size):
        cur_batch = [list(e) for e in cur_batch]
        max_seq_len = 0
        for e in cur_batch:
            max_seq_len = max(max_seq_len, len(e[0]))
        last_id = []
        for e in cur_batch:
            last_id.append(len(e[0]) - 1)
            e[0] += [0] * (max_seq_len - len(e[0]))

        max_uniq_len = 0
        for e in cur_batch:
            max_uniq_len = max(max_uniq_len, len(np.unique(e[0])))

        items, adj_in, adj_out, seq_index, last_index = [], [], [], [], []
        mask, label = [], []

        id = 0
        for e in cur_batch:
            node = np.unique(e[0])
            items.append(node.tolist() + (max_uniq_len - len(node)) * [0])
            adj = np.zeros((max_uniq_len, max_uniq_len))

            for i in np.arange(len(e[0]) - 1):
                if e[0][i + 1] == 0:
                    break
                u = np.where(node == e[0][i])[0][0]
                v = np.where(node == e[0][i + 1])[0][0]
                adj[u][v] = 1

            u_deg_in = np.sum(adj, 0)
            u_deg_in[np.where(u_deg_in == 0)] = 1
            adj_in.append(np.divide(adj, u_deg_in).transpose())

            u_deg_out = np.sum(adj, 1)
            u_deg_out[np.where(u_deg_out == 0)] = 1
            adj_out.append(np.divide(adj.transpose(), u_deg_out).transpose())

            seq_index.append(
                [np.where(node == i)[0][0] + id * max_uniq_len for i in e[0]])
            last_index.append(
                np.where(node == e[0][last_id[id]])[0][0] + id * max_uniq_len)
            label.append(e[1] - 1)
            mask.append([[1] * (last_id[id] + 1) + [0] *
                         (max_seq_len - last_id[id] - 1)])
            id += 1

        items = np.array(items).astype("uint64").reshape((batch_size, -1, 1))
        seq_index = np.array(seq_index).astype("uint64").reshape(
            (batch_size, -1))
        last_index = np.array(last_index).astype("uint64").reshape(
            (batch_size, 1))
        adj_in = np.array(adj_in).astype("float32").reshape(
            (batch_size, max_uniq_len, max_uniq_len))
        adj_out = np.array(adj_out).astype("float32").reshape(
            (batch_size, max_uniq_len, max_uniq_len))
        mask = np.array(mask).astype("float32").reshape((batch_size, -1, 1))
        label = np.array(label).astype("uint64").reshape((batch_size, 1))
        return list(
            zip(items, seq_index, last_index, adj_in, adj_out, mask, label))

    def reader(self, batch_size, batch_group_size, train=True):
        if self.shuffle:
            random.shuffle(self.input)
        group_remain = self.length % batch_group_size
        for bg_id in range(0, self.length - group_remain, batch_group_size):
            cur_bg = copy.deepcopy(self.input[bg_id:bg_id + batch_group_size])
            if train:
                cur_bg = sorted(cur_bg, key=lambda x: len(x[0]), reverse=True)
            for i in range(0, batch_group_size, batch_size):
                cur_batch = cur_bg[i:i + batch_size]
                yield self.make_data(cur_batch, batch_size)

        #deal with the remaining, discard at most batch_size data
        if group_remain < batch_size:
            return
        remain_data = copy.deepcopy(self.input[-group_remain:])
        if train:
            remain_data = sorted(
                remain_data, key=lambda x: len(x[0]), reverse=True)
        for i in range(0, batch_group_size, batch_size):
            if i + batch_size <= len(remain_data):
                cur_batch = remain_data[i:i + batch_size]
                yield self.make_data(cur_batch, batch_size)


def read_config(path):
    with open(path, "r") as fin:
        item_num = int(fin.readline())
    return item_num


induce_map = {0: [0], 1: [0], 2: [], 3: [0, 1], 4: [0, 1], 5: [0], 6: []}


def binary_print(slot, fout, index):
    shape_array = slot.shape
    num = 1
    for e in shape_array:
        num *= e
    num += len(induce_map[index])
    num = np.uint16(num)
    num.tofile(fout)
    for e in induce_map[index]:
        tmp_shape = np.uint64(shape_array[e])
        tmp_shape.tofile(fout)
    slot.tofile(fout)


def make_binary_data():
    data_reader = Data('./data/diginetica/train.txt', True)
    index = 0
    id = -1
    filename = None
    fout = None
    binary = True
    for data in data_reader.reader(batch_size, 20 * batch_size, True):
        if index % (batch_size * 900) == 0:
            id += 1
            if not binary:
                filename = "./gnn_data_text/" + str(id)
            else:
                filename = "./gnn_data_new_8/" + str(id)
            print("filename: " + filename)
            if fout:
                fout.close()
            fout = open(filename, "wb" if binary else "w")

        for ins in data:
            for i, slot in enumerate(ins):
                if binary:
                    binary_print(slot, fout, i)
                else:
                    text_print(slot, fout, i)
        index += batch_size


if __name__ == "__main__":
    make_binary_data()
    train()
