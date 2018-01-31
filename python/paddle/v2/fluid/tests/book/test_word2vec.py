#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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

import paddle.v2 as paddle
import paddle.v2.fluid as fluid
import unittest
import os


def main(use_cuda, is_sparse, parallel):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return

    PASS_NUM = 100
    EMBED_SIZE = 32
    HIDDEN_SIZE = 256
    N = 5
    BATCH_SIZE = 32
    IS_SPARSE = is_sparse

    def __network__(words):
        embed_first = fluid.layers.embedding(
            input=words[0],
            size=[dict_size, EMBED_SIZE],
            dtype='float32',
            is_sparse=IS_SPARSE,
            param_attr='shared_w')
        embed_second = fluid.layers.embedding(
            input=words[1],
            size=[dict_size, EMBED_SIZE],
            dtype='float32',
            is_sparse=IS_SPARSE,
            param_attr='shared_w')
        embed_third = fluid.layers.embedding(
            input=words[2],
            size=[dict_size, EMBED_SIZE],
            dtype='float32',
            is_sparse=IS_SPARSE,
            param_attr='shared_w')
        embed_forth = fluid.layers.embedding(
            input=words[3],
            size=[dict_size, EMBED_SIZE],
            dtype='float32',
            is_sparse=IS_SPARSE,
            param_attr='shared_w')

        concat_embed = fluid.layers.concat(
            input=[embed_first, embed_second, embed_third, embed_forth], axis=1)
        hidden1 = fluid.layers.fc(input=concat_embed,
                                  size=HIDDEN_SIZE,
                                  act='sigmoid')
        predict_word = fluid.layers.fc(input=hidden1,
                                       size=dict_size,
                                       act='softmax')
        cost = fluid.layers.cross_entropy(input=predict_word, label=words[4])
        avg_cost = fluid.layers.mean(x=cost)
        return avg_cost

    word_dict = paddle.dataset.imikolov.build_dict()
    dict_size = len(word_dict)

    first_word = fluid.layers.data(name='firstw', shape=[1], dtype='int64')
    second_word = fluid.layers.data(name='secondw', shape=[1], dtype='int64')
    third_word = fluid.layers.data(name='thirdw', shape=[1], dtype='int64')
    forth_word = fluid.layers.data(name='forthw', shape=[1], dtype='int64')
    next_word = fluid.layers.data(name='nextw', shape=[1], dtype='int64')

    if not parallel:
        avg_cost = __network__(
            [first_word, second_word, third_word, forth_word, next_word])
    else:
        places = fluid.layers.get_places()
        pd = fluid.layers.ParallelDo(places)
        with pd.do():
            avg_cost = __network__(
                map(pd.read_input, [
                    first_word, second_word, third_word, forth_word, next_word
                ]))
            pd.write_output(avg_cost)

        avg_cost = fluid.layers.mean(x=pd())

    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
    sgd_optimizer.minimize(avg_cost)

    train_reader = paddle.batch(
        paddle.dataset.imikolov.train(word_dict, N), BATCH_SIZE)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    feeder = fluid.DataFeeder(
        feed_list=[first_word, second_word, third_word, forth_word, next_word],
        place=place)

    exe.run(fluid.default_startup_program())

    for pass_id in range(PASS_NUM):
        for data in train_reader():
            avg_cost_np = exe.run(fluid.default_main_program(),
                                  feed=feeder.feed(data),
                                  fetch_list=[avg_cost])
            if avg_cost_np[0] < 5.0:
                return
    raise AssertionError("Cost is too large {0:2.2}".format(avg_cost_np[0]))


FULL_TEST = os.getenv('FULL_TEST',
                      '0').lower() in ['true', '1', 't', 'y', 'yes', 'on']
SKIP_REASON = "Only run minimum number of tests in CI server, to make CI faster"


class W2VTest(unittest.TestCase):
    pass


def inject_test_method(use_cuda, is_sparse, parallel):
    fn_name = "test_{0}_{1}_{2}".format("cuda" if use_cuda else "cpu", "sparse"
                                        if is_sparse else "dense", "parallel"
                                        if parallel else "normal")

    def __impl__(*args, **kwargs):
        prog = fluid.Program()
        startup_prog = fluid.Program()
        scope = fluid.core.Scope()
        with fluid.scope_guard(scope):
            with fluid.program_guard(prog, startup_prog):
                main(use_cuda=use_cuda, is_sparse=is_sparse, parallel=parallel)

    if use_cuda and is_sparse and parallel:
        fn = __impl__
    else:
        # skip the other test when on CI server
        fn = unittest.skipUnless(
            condition=FULL_TEST, reason=SKIP_REASON)(__impl__)

    setattr(W2VTest, fn_name, fn)


for use_cuda in (False, True):
    for is_sparse in (False, True):
        for parallel in (False, True):
            inject_test_method(use_cuda, is_sparse, parallel)

if __name__ == '__main__':
    unittest.main()
