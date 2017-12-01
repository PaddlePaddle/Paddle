import numpy as np
import paddle.v2 as paddle
import paddle.v2.fluid as fluid

PASS_NUM = 100
EMBED_SIZE = 32
HIDDEN_SIZE = 256
N = 5
BATCH_SIZE = 32
IS_SPARSE = True

word_dict = paddle.dataset.imikolov.build_dict()
dict_size = len(word_dict)

first_word = fluid.layers.data(name='firstw', shape=[1], dtype='int64')
second_word = fluid.layers.data(name='secondw', shape=[1], dtype='int64')
third_word = fluid.layers.data(name='thirdw', shape=[1], dtype='int64')
forth_word = fluid.layers.data(name='forthw', shape=[1], dtype='int64')
next_word = fluid.layers.data(name='nextw', shape=[1], dtype='int64')

embed_first = fluid.layers.embedding(
    input=first_word,
    size=[dict_size, EMBED_SIZE],
    dtype='float32',
    is_sparse=IS_SPARSE,
    param_attr='shared_w')
embed_second = fluid.layers.embedding(
    input=second_word,
    size=[dict_size, EMBED_SIZE],
    dtype='float32',
    is_sparse=IS_SPARSE,
    param_attr='shared_w')
embed_third = fluid.layers.embedding(
    input=third_word,
    size=[dict_size, EMBED_SIZE],
    dtype='float32',
    is_sparse=IS_SPARSE,
    param_attr='shared_w')
embed_forth = fluid.layers.embedding(
    input=forth_word,
    size=[dict_size, EMBED_SIZE],
    dtype='float32',
    is_sparse=IS_SPARSE,
    param_attr='shared_w')

concat_embed = fluid.layers.concat(
    input=[embed_first, embed_second, embed_third, embed_forth], axis=1)
hidden1 = fluid.layers.fc(input=concat_embed, size=HIDDEN_SIZE, act='sigmoid')
predict_word = fluid.layers.fc(input=hidden1, size=dict_size, act='softmax')
cost = fluid.layers.cross_entropy(input=predict_word, label=next_word)
avg_cost = fluid.layers.mean(x=cost)
sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
sgd_optimizer.minimize(avg_cost)

train_reader = paddle.batch(
    paddle.dataset.imikolov.train(word_dict, N), BATCH_SIZE)

place = fluid.CPUPlace()
exe = fluid.Executor(place)

exe.run(fluid.default_startup_program())

for pass_id in range(PASS_NUM):
    for data in train_reader():
        input_data = [[data_idx[idx] for data_idx in data] for idx in xrange(5)]
        input_data = map(lambda x: np.array(x).astype("int64"), input_data)
        input_data = map(lambda x: np.expand_dims(x, axis=1), input_data)

        avg_cost_np = exe.run(fluid.default_main_program(),
                              feed={
                                  'firstw': input_data[0],
                                  'secondw': input_data[1],
                                  'thirdw': input_data[2],
                                  'forthw': input_data[3],
                                  'nextw': input_data[4]
                              },
                              fetch_list=[avg_cost])
        if avg_cost_np[0] < 5.0:
            exit(0)  # if avg cost less than 10.0, we think our code is good.
exit(1)
