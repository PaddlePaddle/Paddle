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

import paddle
import paddle.fluid as fluid
from paddle.fluid.layers.device import get_places
import unittest
import os
import numpy as np
import math
import sys
import tempfile

paddle.enable_static()


def get_place(target):
    if target == "cuda":
        return fluid.CUDAPlace(0)
    elif target == "xpu":
        return fluid.XPUPlace(0)
    elif target == "cpu":
        return fluid.CPUPlace()
    else:
        raise ValueError(
            "Target `{0}` is not on the support list: `cuda`, `xpu` and `cpu`.".
            format(target))


def train(target,
          is_sparse,
          is_parallel,
          save_dirname,
          is_local=True,
          use_bf16=False,
          pure_bf16=False):
    PASS_NUM = 100
    EMBED_SIZE = 32
    HIDDEN_SIZE = 256
    N = 5
    BATCH_SIZE = 32
    IS_SPARSE = is_sparse

    def __network__(words):
        embed_first = fluid.layers.embedding(input=words[0],
                                             size=[dict_size, EMBED_SIZE],
                                             dtype='float32',
                                             is_sparse=IS_SPARSE,
                                             param_attr='shared_w')
        embed_second = fluid.layers.embedding(input=words[1],
                                              size=[dict_size, EMBED_SIZE],
                                              dtype='float32',
                                              is_sparse=IS_SPARSE,
                                              param_attr='shared_w')
        embed_third = fluid.layers.embedding(input=words[2],
                                             size=[dict_size, EMBED_SIZE],
                                             dtype='float32',
                                             is_sparse=IS_SPARSE,
                                             param_attr='shared_w')
        embed_forth = fluid.layers.embedding(input=words[3],
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
        avg_cost = paddle.mean(cost)
        return avg_cost, predict_word

    word_dict = paddle.dataset.imikolov.build_dict()
    dict_size = len(word_dict)

    first_word = fluid.layers.data(name='firstw', shape=[1], dtype='int64')
    second_word = fluid.layers.data(name='secondw', shape=[1], dtype='int64')
    third_word = fluid.layers.data(name='thirdw', shape=[1], dtype='int64')
    forth_word = fluid.layers.data(name='forthw', shape=[1], dtype='int64')
    next_word = fluid.layers.data(name='nextw', shape=[1], dtype='int64')

    if not is_parallel:
        avg_cost, predict_word = __network__(
            [first_word, second_word, third_word, forth_word, next_word])
    else:
        raise NotImplementedError()

    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
    if use_bf16:
        sgd_optimizer = paddle.static.amp.bf16.decorate_bf16(
            sgd_optimizer,
            amp_lists=paddle.static.amp.bf16.AutoMixedPrecisionListsBF16(
                custom_fp32_list={'softmax', 'concat'}, ),
            use_bf16_guard=False,
            use_pure_bf16=pure_bf16)

    sgd_optimizer.minimize(avg_cost, fluid.default_startup_program())

    train_reader = paddle.batch(paddle.dataset.imikolov.train(word_dict, N),
                                BATCH_SIZE)

    place = get_place(target)
    exe = fluid.Executor(place)
    feeder = fluid.DataFeeder(
        feed_list=[first_word, second_word, third_word, forth_word, next_word],
        place=place)

    def train_loop(main_program):
        exe.run(fluid.default_startup_program())
        if pure_bf16:
            sgd_optimizer.amp_init(exe.place)

        for pass_id in range(PASS_NUM):
            for data in train_reader():
                avg_cost_np = exe.run(main_program,
                                      feed=feeder.feed(data),
                                      fetch_list=[avg_cost])
                if avg_cost_np[0] < 5.0:
                    if save_dirname is not None and not pure_bf16:
                        fluid.io.save_inference_model(
                            save_dirname,
                            ['firstw', 'secondw', 'thirdw', 'forthw'],
                            [predict_word], exe)
                    return
                if math.isnan(float(avg_cost_np[0])):
                    sys.exit("got NaN loss, training failed.")

        raise AssertionError("Cost is too large {0:2.2}".format(avg_cost_np[0]))

    if is_local:
        train_loop(fluid.default_main_program())
    else:
        port = os.getenv("PADDLE_PSERVER_PORT", "6174")
        pserver_ips = os.getenv("PADDLE_PSERVER_IPS")  # ip,ip...
        eplist = []
        for ip in pserver_ips.split(","):
            eplist.append(':'.join([ip, port]))
        pserver_endpoints = ",".join(eplist)  # ip:port,ip:port...
        trainers = int(os.getenv("PADDLE_TRAINERS"))
        current_endpoint = os.getenv("POD_IP") + ":" + port
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID"))
        training_role = os.getenv("PADDLE_TRAINING_ROLE", "TRAINER")
        t = fluid.DistributeTranspiler()
        t.transpile(trainer_id, pservers=pserver_endpoints, trainers=trainers)
        if training_role == "PSERVER":
            pserver_prog = t.get_pserver_program(current_endpoint)
            pserver_startup = t.get_startup_program(current_endpoint,
                                                    pserver_prog)
            exe.run(pserver_startup)
            exe.run(pserver_prog)
        elif training_role == "TRAINER":
            train_loop(t.get_trainer_program())


def infer(target, save_dirname=None):
    if save_dirname is None:
        return

    place = get_place(target)
    exe = fluid.Executor(place)
    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        # Use fluid.io.load_inference_model to obtain the inference program desc,
        # the feed_target_names (the names of variables that will be fed
        # data using feed operators), and the fetch_targets (variables that
        # we want to obtain data from using fetch operators).
        [inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(save_dirname, exe)

        word_dict = paddle.dataset.imikolov.build_dict()
        dict_size = len(word_dict)

        # Setup inputs by creating 4 LoDTensors representing 4 words. Here each word
        # is simply an index to look up for the corresponding word vector and hence
        # the shape of word (base_shape) should be [1]. The recursive_sequence_lengths,
        # which is length-based level of detail (lod) of each LoDTensor, should be [[1]]
        # meaning there is only one level of detail and there is only one sequence of
        # one word on this level.
        # Note that recursive_sequence_lengths should be a list of lists.
        recursive_seq_lens = [[1]]
        base_shape = [1]
        # The range of random integers is [low, high]
        first_word = fluid.create_random_int_lodtensor(recursive_seq_lens,
                                                       base_shape,
                                                       place,
                                                       low=0,
                                                       high=dict_size - 1)
        second_word = fluid.create_random_int_lodtensor(recursive_seq_lens,
                                                        base_shape,
                                                        place,
                                                        low=0,
                                                        high=dict_size - 1)
        third_word = fluid.create_random_int_lodtensor(recursive_seq_lens,
                                                       base_shape,
                                                       place,
                                                       low=0,
                                                       high=dict_size - 1)
        fourth_word = fluid.create_random_int_lodtensor(recursive_seq_lens,
                                                        base_shape,
                                                        place,
                                                        low=0,
                                                        high=dict_size - 1)

        assert feed_target_names[0] == 'firstw'
        assert feed_target_names[1] == 'secondw'
        assert feed_target_names[2] == 'thirdw'
        assert feed_target_names[3] == 'forthw'

        # Construct feed as a dictionary of {feed_target_name: feed_target_data}
        # and results will contain a list of data corresponding to fetch_targets.
        results = exe.run(inference_program,
                          feed={
                              feed_target_names[0]: first_word,
                              feed_target_names[1]: second_word,
                              feed_target_names[2]: third_word,
                              feed_target_names[3]: fourth_word
                          },
                          fetch_list=fetch_targets,
                          return_numpy=False)

        def to_infer_tensor(lod_tensor):
            infer_tensor = fluid.core.PaddleTensor()
            infer_tensor.lod = lod_tensor.lod()
            infer_tensor.data = fluid.core.PaddleBuf(np.array(lod_tensor))
            infer_tensor.shape = lod_tensor.shape()
            infer_tensor.dtype = fluid.core.PaddleDType.INT64
            return infer_tensor

        infer_inputs = [first_word, second_word, third_word, fourth_word]
        infer_inputs = [to_infer_tensor(t) for t in infer_inputs]

        infer_config = fluid.core.NativeConfig()
        infer_config.model_dir = save_dirname
        if target == "cuda":
            infer_config.use_gpu = True
            infer_config.device = 0
            infer_config.fraction_of_gpu_memory = 0.15
        elif target == "xpu":
            infer_config.use_xpu = True
        compiled_program = fluid.compiler.CompiledProgram(inference_program)
        compiled_program._with_inference_optimize(infer_config)
        assert compiled_program._is_inference is True
        infer_outputs = exe.run(compiled_program, feed=infer_inputs)
        np_data = np.array(results[0])
        infer_out = infer_outputs[0].data.float_data()
        for a, b in zip(np_data[0], infer_out):
            assert np.isclose(a, b, rtol=5e-5), "a: {}, b: {}".format(a, b)


def main(target, is_sparse, is_parallel, use_bf16, pure_bf16):
    if target == "cuda" and not fluid.core.is_compiled_with_cuda():
        return
    if target == "xpu" and not fluid.core.is_compiled_with_xpu():
        return

    if use_bf16 and not fluid.core.is_compiled_with_mkldnn():
        return

    temp_dir = tempfile.TemporaryDirectory()
    if not is_parallel:
        save_dirname = os.path.join(temp_dir.name, "word2vec.inference.model")
    else:
        save_dirname = None

    if target == "xpu":
        # This model cannot be trained with xpu temporarily,
        # so only inference is turned on.
        train("cpu", is_sparse, is_parallel, save_dirname)
    else:
        train(target,
              is_sparse,
              is_parallel,
              save_dirname,
              use_bf16=use_bf16,
              pure_bf16=pure_bf16)
    infer(target, save_dirname)
    temp_dir.cleanup()


FULL_TEST = os.getenv('FULL_TEST',
                      '0').lower() in ['true', '1', 't', 'y', 'yes', 'on']
SKIP_REASON = "Only run minimum number of tests in CI server, to make CI faster"


class W2VTest(unittest.TestCase):
    pass


def inject_test_method(target,
                       is_sparse,
                       is_parallel,
                       use_bf16=False,
                       pure_bf16=False):
    fn_name = "test_{0}_{1}_{2}{3}".format(
        target, "sparse" if is_sparse else "dense",
        "parallel" if is_parallel else "normal",
        "_purebf16" if pure_bf16 else "_bf16" if use_bf16 else "")

    def __impl__(*args, **kwargs):
        prog = fluid.Program()
        startup_prog = fluid.Program()
        scope = fluid.core.Scope()
        with fluid.scope_guard(scope):
            with fluid.program_guard(prog, startup_prog):
                main(target, is_sparse, is_parallel, use_bf16, pure_bf16)

    if (not fluid.core.is_compiled_with_cuda()
            or target == "cuda") and is_sparse:
        fn = __impl__
    else:
        # skip the other test when on CI server
        fn = unittest.skipUnless(condition=FULL_TEST,
                                 reason=SKIP_REASON)(__impl__)

    setattr(W2VTest, fn_name, fn)


for target in ("cuda", "cpu", "xpu"):
    for is_sparse in (False, True):
        for is_parallel in (False, ):
            inject_test_method(target, is_sparse, is_parallel)
inject_test_method("cpu", False, False, True)
inject_test_method("cpu", False, False, True, True)

if __name__ == '__main__':
    unittest.main()
