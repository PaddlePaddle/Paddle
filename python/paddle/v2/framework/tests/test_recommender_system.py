import paddle.v2 as paddle
import paddle.v2.framework.layers as layers
import paddle.v2.framework.nets as nets
import paddle.v2.framework.core as core
import paddle.v2.framework.optimizer as optimizer

from paddle.v2.framework.framework import Program, g_program
from paddle.v2.framework.executor import Executor

import numpy as np

init_program = Program()
program = Program()


def get_usr_combined_features():
    # FIXME(dzh) : old API integer_value(10) may has range check.
    # currently we don't have user configurated check.

    USR_DICT_SIZE = paddle.dataset.movielens.max_user_id() + 1

    uid = layers.data(
        name='user_id',
        shape=[1],
        data_type='int32',
        program=program,
        init_program=init_program)

    usr_emb = layers.embedding(
        input=uid,
        size=[USR_DICT_SIZE, 32],
        param_attr={'name': 'user_table'},
        program=program,
        init_program=init_program)

    usr_fc = layers.fc(input=usr_emb,
                       size=32,
                       program=program,
                       init_program=init_program)

    USR_GENDER_DICT_SIZE = 2

    usr_gender_id = layers.data(
        name='gender_id',
        shape=[1],
        data_type='int32',
        program=program,
        init_program=init_program)

    usr_gender_emb = layers.embedding(
        input=usr_gender_id,
        size=[USR_GENDER_DICT_SIZE, 16],
        param_attr={'name': 'gender_table'},
        program=program,
        init_program=init_program)

    usr_gender_fc = layers.fc(input=usr_gender_emb,
                              size=16,
                              program=program,
                              init_program=init_program)

    USR_AGE_DICT_SIZE = len(paddle.dataset.movielens.age_table)
    usr_age_id = layers.data(
        name='age_id',
        shape=[1],
        data_type="int32",
        program=program,
        init_program=init_program)

    usr_age_emb = layers.embedding(
        input=usr_age_id,
        size=[USR_AGE_DICT_SIZE, 16],
        param_attr={'name': 'age_table'},
        program=program,
        init_program=init_program)

    usr_age_fc = layers.fc(input=usr_age_emb,
                           size=16,
                           program=program,
                           init_program=init_program)

    USR_JOB_DICT_SIZE = paddle.dataset.movielens.max_job_id() + 1
    usr_job_id = layers.data(
        name='job_id',
        shape=[1],
        data_type="int32",
        program=program,
        init_program=init_program)

    usr_job_emb = layers.embedding(
        input=usr_job_id,
        size=[USR_JOB_DICT_SIZE, 16],
        param_attr={'name': 'job_table'},
        program=program,
        init_program=init_program)

    usr_job_fc = layers.fc(input=usr_job_emb,
                           size=16,
                           program=program,
                           init_program=init_program)

    concat_embed = layers.concat(
        input=[usr_fc, usr_gender_fc, usr_age_fc, usr_job_fc],
        axis=1,
        program=program,
        init_program=init_program)

    # FIXME(dzh) : need tanh operator
    usr_combined_features = layers.fc(input=concat_embed,
                                      size=200,
                                      act="tanh",
                                      program=program,
                                      init_program=init_program)

    return usr_combined_features


def get_mov_combined_features():

    MOV_DICT_SIZE = paddle.dataset.movielens.max_movie_id() + 1

    mov_id = layers.data(
        name='movie_id',
        shape=[1],
        data_type='int32',
        program=program,
        init_program=init_program)

    mov_emb = layers.embedding(
        input=mov_id,
        size=[MOV_DICT_SIZE, 32],
        param_attr={'name': 'movie_table'},
        program=program,
        init_program=init_program)

    mov_fc = layers.fc(input=mov_emb,
                       size=32,
                       program=program,
                       init_program=init_program)

    CATEGORY_DICT_SIZE = len(paddle.dataset.movielens.movie_categories())

    category_id = layers.data(
        name='category_id',
        shape=[1],
        data_type='int32',
        program=program,
        init_program=init_program)

    mov_categories_emb = layers.embedding(
        input=category_id,
        size=[CATEGORY_DICT_SIZE, 32],
        program=program,
        init_program=init_program)

    mov_categories_hidden = layers.sequence_pool(
        input=mov_categories_emb,
        pool_size=None,
        pool_type="sum",
        program=program,
        init_program=init_program)

    MOV_TITLE_DICT_SIZE = len(paddle.dataset.movielens.get_movie_title_dict())

    mov_title_id = layers.data(
        name='movie_title',
        shape=[1],
        data_type='int32',
        program=program,
        init_program=init_program)

    mov_title_emb = layers.embedding(
        input=mov_title_id,
        size=[MOV_TITLE_DICT_SIZE, 32],
        program=program,
        init_program=init_program)

    mov_title_conv = nets.sequence_conv_pool(
        input=mov_title_emb,
        num_filters=32,
        filter_size=3,
        pool_type="sum",
        pool_size=3,
        pool_stride=1,
        act="tanh",
        program=program,
        init_program=init_program)

    concat_embed = layers.concat(
        input=[mov_fc, mov_categories_hidden, mov_title_conv],
        axis=1,
        program=program,
        init_program=init_program)

    # FIXME(dzh) : need tanh operator
    mov_combined_features = layers.fc(input=concat_embed,
                                      size=200,
                                      act="tanh",
                                      program=program,
                                      init_program=init_program)

    return mov_combined_features


def model():
    usr_combined_features = get_usr_combined_features()
    mov_combined_features = get_mov_combined_features()

    # need cos sim
    inference, _, _ = layers.cos_sim(
        X=usr_combined_features,
        Y=mov_combined_features,
        program=program,
        init_program=init_program)

    label = layers.data(
        name='score',
        shape=[1],
        data_type='int32',
        program=program,
        init_program=init_program)

    cost = layers.square_error_cost(
        input=inference,
        label=label,
        program=program,
        init_program=init_program)

    return cost


def main():
    cost = model()
    adam_optimizer = optimizer.AdamOptimizer(learning_rate=1e-4)
    opts = adam_optimizer.minimize(cost)

    place = core.CPUPlace()
    exe = Executor(place)
    exe.run(init_program, feed={}, fetch_list=[])

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.movielens.train(), buf_size=8192),
        batch_size=256)

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

    def func_feed(feeding, data):
        feed_tensors = {}
        for (key, idx) in feeding.iteritems():
            tensor = core.LoDTensor()
            if key != "category_id" and key != "movie_title":
                numpy_data = np.array(map(lambda x: x[idx], data)).astype(
                    "int32")
            else:
                numpy_data = map(lambda x: np.array(x[idx]).astype("int32"),
                                 data)
                lod_info = [len(item) for item in numpy_data]
                offset = 0
                lod = [offset]
                for item in lod_info:
                    offset += item
                    lod.append(offset)
                numpy_data = np.concatenate(numpy_data, axis=0)
                tensor.set_lod([lod])

            tensor.set(numpy_data, place)
            feed_tensors[key] = tensor
        return feed_tensors

    PASS_NUM = 100
    for pass_id in range(PASS_NUM):
        for data in train_reader():
            print data
            print func_feed(feeding, data)
            outs = exe.run(program,
                           feed=func_feed(feeding, data),
                           fetch_list=[cost])
            out = np.array(outs[0])
            if out[0] < 10.0:
                exit(
                    0)  # if avg cost less than 10.0, we think our code is good.
            else:
                exit(1)


main()
