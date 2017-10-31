import paddle.v2 as paddle
import paddle.v2.framework.layers as layers
import paddle.v2.framework.core as core
import paddle.v2.framework.optimizer as optimizer

from paddle.v2.framework.framework import Program, g_program
from paddle.v2.framework.executor import Executor
from paddle.v2.framework.initializer import NormalInitializer

import numpy as np
import math

init_program = Program()
program = Program()

embed_size = 32
hidden_size = 256
N = 5
batch_size = 32
is_sparse = False

word_dict = paddle.dataset.imikolov.build_dict()
dict_size = len(word_dict)

first_word = layers.data(
    name='firstw',
    shape=[1],
    data_type='int64',
    program=program,
    init_program=init_program)
second_word = layers.data(
    name='secondw',
    shape=[1],
    data_type='int64',
    program=program,
    init_program=init_program)
third_word = layers.data(
    name='thirdw',
    shape=[1],
    data_type='int64',
    program=program,
    init_program=init_program)
forth_word = layers.data(
    name='forthw',
    shape=[1],
    data_type='int64',
    program=program,
    init_program=init_program)
next_word = layers.data(
    name='nextw',
    shape=[1],
    data_type='int64',
    program=program,
    init_program=init_program)

embed_param_init = NormalInitializer(std=0.001)

embed_param_attr = {'name': 'shared_w', 'initializer': embed_param_init}
# the shared param attr should not have initializer
embed_param_attr_shared = {
    'name': 'shared_w',
    'is_shared': True,
}

embed_first = layers.embedding(
    input=first_word,
    size=[dict_size, embed_size],
    data_type='float32',
    is_sparse=is_sparse,
    param_attr=embed_param_attr,
    program=program,
    init_program=init_program)
embed_second = layers.embedding(
    input=second_word,
    size=[dict_size, embed_size],
    data_type='float32',
    is_sparse=is_sparse,
    param_attr=embed_param_attr_shared,
    program=program,
    init_program=init_program)

embed_third = layers.embedding(
    input=third_word,
    size=[dict_size, embed_size],
    data_type='float32',
    is_sparse=is_sparse,
    param_attr=embed_param_attr_shared,
    program=program,
    init_program=init_program)
embed_forth = layers.embedding(
    input=forth_word,
    size=[dict_size, embed_size],
    data_type='float32',
    is_sparse=is_sparse,
    param_attr=embed_param_attr_shared,
    program=program,
    init_program=init_program)

concat_embed = layers.concat(
    input=[embed_first, embed_second, embed_third, embed_forth],
    axis=1,
    program=program,
    init_program=init_program)

hidden1_param_init = NormalInitializer(std=1 / math.sqrt(embed_size * 8))
hidden1_param_attr = {'initializer': hidden1_param_init}
hidden1_bias_attr = {'optimize_attr': {'learning_rate': 2.0}}

# TODO(qijun) need to add dropout layer
hidden1 = layers.fc(input=concat_embed,
                    size=hidden_size,
                    param_attr=hidden1_param_attr,
                    bias_attr=hidden1_bias_attr,
                    act='sigmoid',
                    program=program,
                    init_program=init_program)

predict_bias_attr = {'optimize_attr': {'learning_rate': 2.0}}
predict_word = layers.fc(input=hidden1,
                         size=dict_size,
                         act='softmax',
                         program=program,
                         init_program=init_program)
cost = layers.cross_entropy(
    input=predict_word,
    label=next_word,
    program=program,
    init_program=init_program)
avg_cost = layers.mean(x=cost, program=program, init_program=init_program)

adagrad_optimizer = optimizer.AdagradOptimizer(learning_rate=0.001)
opts = adagrad_optimizer.minimize(avg_cost)

train_reader = paddle.batch(
    paddle.dataset.imikolov.train(word_dict, N), batch_size)

place = core.CPUPlace()
exe = Executor(place)

exe.run(init_program, feed={}, fetch_list=[])
PASS_NUM = 30
for pass_id in range(PASS_NUM):
    # print 'pass: ', pass_id
    for data in train_reader():
        input_data = [[data_idx[idx] for data_idx in data] for idx in xrange(5)]
        input_data = map(lambda x: np.array(x).astype("int64"), input_data)
        input_data = map(lambda x: np.expand_dims(x, axis=1), input_data)

        first_data = input_data[0]
        first_tensor = core.LoDTensor()
        first_tensor.set(first_data, place)

        second_data = input_data[1]
        second_tensor = core.LoDTensor()
        second_tensor.set(second_data, place)

        third_data = input_data[2]
        third_tensor = core.LoDTensor()
        third_tensor.set(third_data, place)

        forth_data = input_data[3]
        forth_tensor = core.LoDTensor()
        forth_tensor.set(forth_data, place)

        next_data = input_data[4]
        next_tensor = core.LoDTensor()
        next_tensor.set(next_data, place)

        outs = exe.run(program,
                       feed={
                           'firstw': first_tensor,
                           'secondw': second_tensor,
                           'thirdw': third_tensor,
                           'forthw': forth_tensor,
                           'nextw': next_tensor
                       },
                       fetch_list=[avg_cost])
        loss = np.array(outs[0])
        # print loss
        if loss[0] < 5.0:
            exit(0)  # if avg cost less than 5.0, we think our code is good.
exit(1)
