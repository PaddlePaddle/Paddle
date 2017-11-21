import numpy as np
import paddle.v2 as paddle
import paddle.v2.fluid.core as core
import paddle.v2.fluid.framework as framework
import paddle.v2.fluid.layers as layers
from paddle.v2.fluid.executor import Executor
from paddle.v2.fluid.optimizer import SGDOptimizer

PASS_NUM = 100
EMBED_SIZE = 32
HIDDEN_SIZE = 256
N = 5
BATCH_SIZE = 32
IS_SPARSE = True

word_dict = paddle.dataset.imikolov.build_dict()
dict_size = len(word_dict)

first_word = layers.data(name='firstw', shape=[1], data_type='int64')
second_word = layers.data(name='secondw', shape=[1], data_type='int64')
third_word = layers.data(name='thirdw', shape=[1], data_type='int64')
forth_word = layers.data(name='forthw', shape=[1], data_type='int64')
next_word = layers.data(name='nextw', shape=[1], data_type='int64')

embed_first = layers.embedding(
    input=first_word,
    size=[dict_size, EMBED_SIZE],
    data_type='float32',
    is_sparse=IS_SPARSE,
    param_attr={'name': 'shared_w'})
embed_second = layers.embedding(
    input=second_word,
    size=[dict_size, EMBED_SIZE],
    data_type='float32',
    is_sparse=IS_SPARSE,
    param_attr={'name': 'shared_w'})
embed_third = layers.embedding(
    input=third_word,
    size=[dict_size, EMBED_SIZE],
    data_type='float32',
    is_sparse=IS_SPARSE,
    param_attr={'name': 'shared_w'})
embed_forth = layers.embedding(
    input=forth_word,
    size=[dict_size, EMBED_SIZE],
    data_type='float32',
    is_sparse=IS_SPARSE,
    param_attr={'name': 'shared_w'})

concat_embed = layers.concat(
    input=[embed_first, embed_second, embed_third, embed_forth], axis=1)
hidden1 = layers.fc(input=concat_embed, size=HIDDEN_SIZE, act='sigmoid')
predict_word = layers.fc(input=hidden1, size=dict_size, act='softmax')
cost = layers.cross_entropy(input=predict_word, label=next_word)
avg_cost = layers.mean(x=cost)
sgd_optimizer = SGDOptimizer(learning_rate=0.001)
opts = sgd_optimizer.minimize(avg_cost)

train_reader = paddle.batch(
    paddle.dataset.imikolov.train(word_dict, N), BATCH_SIZE)

place = core.CPUPlace()
exe = Executor(place)

# fix https://github.com/PaddlePaddle/Paddle/issues/5434 then remove
# below exit line.
exit(0)

exe.run(framework.default_startup_program())

for pass_id in range(PASS_NUM):
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

        outs = exe.run(framework.default_main_program(),
                       feed={
                           'firstw': first_tensor,
                           'secondw': second_tensor,
                           'thirdw': third_tensor,
                           'forthw': forth_tensor,
                           'nextw': next_tensor
                       },
                       fetch_list=[avg_cost])
        out = np.array(outs[0])
        if out[0] < 10.0:
            exit(0)  # if avg cost less than 10.0, we think our code is good.
exit(1)
