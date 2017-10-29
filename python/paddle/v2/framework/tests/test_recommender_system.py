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

    mov_categories_hidden = layers.embedding(
        input=category_id,
        size=[CATEGORY_DICT_SIZE, 32],
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
        input=mov_title_emb, num_filters=32, filter_size=3, pool_size=3, pool_stride=1, act="tanh")

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

    inference = layers.cos_sim(
        a=usr_combined_features,
        b=mov_combined_features,
        size=1, scale=5,
                                      program=program,
                                      init_program=init_program)

    label = layers.data(
        name='score',
        shape=[1],
        data_type='float32',
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

    train_reader=paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.movielens.train(), buf_size=8192),
        batch_size=256),

    place = core.CPUPlace()
    exe = Executor(place)
    exe.run(init_program, feed={}, fetch_list=[])
    PASS_NUM = 100


    for pass_id in range(PASS_NUM):
        for data in train_reader():
            print data

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

    # def event_handler(event):
    #     if isinstance(event, paddle.event.EndIteration):
    #         if event.batch_id % 100 == 0:
    #             print "Pass %d Batch %d Cost %.2f" % (
    #                 event.pass_id, event.batch_id, event.cost)

    # trainer.train(
    #     reader=paddle.batch(
    #         paddle.reader.shuffle(
    #             paddle.dataset.movielens.train(), buf_size=8192),
    #         batch_size=256),
    #     event_handler=event_handler,
    #     feeding=feeding,
    #     num_passes=1)

    # user_id = 234
    # movie_id = 345

    # user = paddle.dataset.movielens.user_info()[user_id]
    # movie = paddle.dataset.movielens.movie_info()[movie_id]

    # feature = user.value() + movie.value()

    # infer_dict = copy.copy(feeding)
    # del infer_dict['score']

    # prediction = paddle.infer(
    #     output_layer=inference,
    #     parameters=parameters,
    #     input=[feature],
    #     feeding=infer_dict)
    # print(prediction + 5) / 2


if __name__ == '__main__':
    main()
