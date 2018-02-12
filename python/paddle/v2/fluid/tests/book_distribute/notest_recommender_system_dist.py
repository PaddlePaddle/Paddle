#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import os
import paddle.v2 as paddle
import paddle.v2.fluid as fluid
import paddle.v2.fluid.core as core
import paddle.v2.fluid.layers as layers
import paddle.v2.fluid.nets as nets
from paddle.v2.fluid.optimizer import SGDOptimizer

IS_SPARSE = True
BATCH_SIZE = 256
PASS_NUM = 100


def get_usr_combined_features():
    USR_DICT_SIZE = paddle.dataset.movielens.max_user_id() + 1
    uid = layers.data(name='user_id', shape=[1], dtype='int64')
    usr_emb = layers.embedding(
        input=uid,
        dtype='float32',
        size=[USR_DICT_SIZE, 32],
        param_attr='user_table',
        is_sparse=IS_SPARSE)
    usr_fc = layers.fc(input=usr_emb, size=32)
    USR_GENDER_DICT_SIZE = 2

    usr_gender_id = layers.data(name='gender_id', shape=[1], dtype='int64')
    usr_gender_emb = layers.embedding(
        input=usr_gender_id,
        size=[USR_GENDER_DICT_SIZE, 16],
        param_attr='gender_table',
        is_sparse=IS_SPARSE)
    usr_gender_fc = layers.fc(input=usr_gender_emb, size=16)

    USR_AGE_DICT_SIZE = len(paddle.dataset.movielens.age_table)
    usr_age_id = layers.data(name='age_id', shape=[1], dtype="int64")
    usr_age_emb = layers.embedding(
        input=usr_age_id,
        size=[USR_AGE_DICT_SIZE, 16],
        is_sparse=IS_SPARSE,
        param_attr='age_table')
    usr_age_fc = layers.fc(input=usr_age_emb, size=16)

    USR_JOB_DICT_SIZE = paddle.dataset.movielens.max_job_id() + 1
    usr_job_id = layers.data(name='job_id', shape=[1], dtype="int64")
    usr_job_emb = layers.embedding(
        input=usr_job_id,
        size=[USR_JOB_DICT_SIZE, 16],
        param_attr='job_table',
        is_sparse=IS_SPARSE)
    usr_job_fc = layers.fc(input=usr_job_emb, size=16)

    concat_embed = layers.concat(
        input=[usr_fc, usr_gender_fc, usr_age_fc, usr_job_fc], axis=1)

    usr_combined_features = layers.fc(input=concat_embed, size=200, act="tanh")
    return usr_combined_features


def get_mov_combined_features():
    MOV_DICT_SIZE = paddle.dataset.movielens.max_movie_id() + 1
    mov_id = layers.data(name='movie_id', shape=[1], dtype='int64')
    mov_emb = layers.embedding(
        input=mov_id,
        dtype='float32',
        size=[MOV_DICT_SIZE, 32],
        param_attr='movie_table',
        is_sparse=IS_SPARSE)
    mov_fc = layers.fc(input=mov_emb, size=32)

    CATEGORY_DICT_SIZE = len(paddle.dataset.movielens.movie_categories())
    category_id = layers.data(name='category_id', shape=[1], dtype='int64')
    mov_categories_emb = layers.embedding(
        input=category_id, size=[CATEGORY_DICT_SIZE, 32], is_sparse=IS_SPARSE)
    mov_categories_hidden = layers.sequence_pool(
        input=mov_categories_emb, pool_type="sum")

    MOV_TITLE_DICT_SIZE = len(paddle.dataset.movielens.get_movie_title_dict())
    mov_title_id = layers.data(name='movie_title', shape=[1], dtype='int64')
    mov_title_emb = layers.embedding(
        input=mov_title_id, size=[MOV_TITLE_DICT_SIZE, 32], is_sparse=IS_SPARSE)
    mov_title_conv = nets.sequence_conv_pool(
        input=mov_title_emb,
        num_filters=32,
        filter_size=3,
        act="tanh",
        pool_type="sum")

    concat_embed = layers.concat(
        input=[mov_fc, mov_categories_hidden, mov_title_conv], axis=1)

    mov_combined_features = layers.fc(input=concat_embed, size=200, act="tanh")
    return mov_combined_features


def model():
    usr_combined_features = get_usr_combined_features()
    mov_combined_features = get_mov_combined_features()

    # need cos sim
    inference = layers.cos_sim(X=usr_combined_features, Y=mov_combined_features)
    scale_infer = layers.scale(x=inference, scale=5.0)

    label = layers.data(name='score', shape=[1], dtype='float32')
    square_cost = layers.square_error_cost(input=scale_infer, label=label)
    avg_cost = layers.mean(x=square_cost)

    return avg_cost


def func_feed(feeding, data, place):
    feed_tensors = {}
    for (key, idx) in feeding.iteritems():
        tensor = core.LoDTensor()
        if key != "category_id" and key != "movie_title":
            if key == "score":
                numpy_data = np.array(map(lambda x: x[idx], data)).astype(
                    "float32")
            else:
                numpy_data = np.array(map(lambda x: x[idx], data)).astype(
                    "int64")
        else:
            numpy_data = map(lambda x: np.array(x[idx]).astype("int64"), data)
            lod_info = [len(item) for item in numpy_data]
            offset = 0
            lod = [offset]
            for item in lod_info:
                offset += item
                lod.append(offset)
            numpy_data = np.concatenate(numpy_data, axis=0)
            tensor.set_lod([lod])

        numpy_data = numpy_data.reshape([numpy_data.shape[0], 1])
        tensor.set(numpy_data, place)
        feed_tensors[key] = tensor
    return feed_tensors


def main():
    cost = model()
    optimizer = SGDOptimizer(learning_rate=0.2)
    optimize_ops, params_grads = optimizer.minimize(cost)

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.movielens.train(), buf_size=8192),
        batch_size=BATCH_SIZE)

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    t = fluid.DistributeTranspiler()

    # all parameter server endpoints list for spliting parameters
    pserver_endpoints = os.getenv("PSERVERS")
    # server endpoint for current node
    current_endpoint = os.getenv("SERVER_ENDPOINT")
    # run as trainer or parameter server
    training_role = os.getenv("TRAINING_ROLE", "TRAINER")
    t.transpile(
        optimize_ops, params_grads, pservers=pserver_endpoints, trainers=2)

    if training_role == "PSERVER":
        if not current_endpoint:
            print("need env SERVER_ENDPOINT")
            exit(1)
        pserver_prog = t.get_pserver_program(current_endpoint)
        pserver_startup = t.get_startup_program(current_endpoint, pserver_prog)
        exe.run(pserver_startup)
        exe.run(pserver_prog)
    elif training_role == "TRAINER":
        exe.run(fluid.default_startup_program())
        trainer_prog = t.get_trainer_program()

        feeding = {
            'user_id': 0,
            'gender_id': 1,
            'age_id': 2,
            'job_id': 3,
            'movie_id': 4,
            'category_id': 5,
            'movie_title': 6,
            'score': 7
        }

        for pass_id in range(PASS_NUM):
            for data in train_reader():
                outs = exe.run(trainer_prog,
                               feed=func_feed(feeding, data, place),
                               fetch_list=[cost])
                out = np.array(outs[0])
                print("cost=" + str(out[0]))
                if out[0] < 6.0:
                    print("Training complete. Average cost is less than 6.0.")
                    # if avg cost less than 6.0, we think our code is good.
                    exit(0)
    else:
        print("environment var TRAINER_ROLE should be TRAINER os PSERVER")


if __name__ == '__main__':
    main()
