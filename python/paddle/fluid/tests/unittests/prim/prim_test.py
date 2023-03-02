#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import random
import unittest
from typing import Sequence

import config
import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid.framework import Variable
from paddle.fluid.layers.utils import map_structure


def _as_list(x):
    if x is None:
        return []
    return list(x) if isinstance(x, Sequence) else [x]


def apply_to_static(net, use_cinn):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(net, build_strategy=build_strategy)


def skip_base_class_test(f):
    def wrapper(*args, **kwargs):
        if args[0].__class__.__name__ is 'PrimTest':
            return
        return f(*args, **kwargs)

    return wrapper


def flatten(nest_list: Sequence):
    out = []
    for i in nest_list:
        if isinstance(i, Sequence):
            tmp_list = flatten(i)
            for j in tmp_list:
                out.append(j)
        else:
            out.append(i)
    return out


def convert_np_list_data_type(np_list, dtype):
    return list(map_structure(lambda x: x.astype(dtype), np_list))


def get_need_grad_xs(inputs, grad_mask):
    xs = []
    for i in range(len(inputs)):
        if isinstance(inputs[i], Sequence):
            xs += (
                get_need_grad_xs(inputs[i], None)
                if grad_mask is None
                else get_need_grad_xs(inputs[i], grad_mask[i])
            )
        else:
            if grad_mask is None or grad_mask[i]:
                xs.append(inputs[i])
    return xs


def gen_static_data_and_feed(np_xs, dtype, base_name):
    feed = {}
    static_xs = []
    if isinstance(np_xs, Sequence):
        for i, x in enumerate(np_xs):
            if isinstance(x, Sequence):
                xs_sub, feed_sub = gen_static_data_and_feed(
                    x, f"{base_name}_{i}"
                )
                static_xs.append(xs_sub)
                feed.update(feed_sub)
            else:
                data = paddle.static.data(f"{base_name}_{i}", x.shape, dtype)
                data.stop_gradient = False
                static_xs.append(data)
                feed.update({f"{base_name}_{i}": x})
    else:
        data = paddle.static.data(f"{base_name}_0", np_xs.shape, dtype)
        data.stop_gradient = False
        static_xs.append(data)
        feed.update({f"{base_name}_0": np_xs})
    return static_xs, feed


def gen_eager_data(np_xs, dtype, place):
    eager_xs = []
    if isinstance(np_xs, Sequence):
        for x in np_xs:
            if isinstance(x, Sequence):
                eager_xs.append(gen_eager_data(x, dtype, place))
            else:
                eager_xs.append(
                    paddle.to_tensor(
                        x, dtype=dtype, place=place, stop_gradient=False
                    )
                )
    else:
        eager_xs.append(
            paddle.to_tensor(
                np_xs, dtype=dtype, place=place, stop_gradient=False
            )
        )
    return eager_xs


def _add_fetch_ops(program, fetch_list, fetch_var_name="fetch"):
    # assert isinstance(program, fluid.Program)
    tmp_program = program.clone()
    global_block = tmp_program.global_block()

    if fetch_var_name in global_block.vars:
        fetch_var = global_block.var(fetch_var_name)
    else:
        fetch_var = global_block.create_var(
            name=fetch_var_name,
            type=core.VarDesc.VarType.FETCH_LIST,
            persistable=True,
        )

    # append fetch_operators
    if not fluid.executor.has_fetch_operators(
        global_block, fetch_list, fetch_var_name, 'fetch'
    ):
        for i, var in enumerate(fetch_list):
            assert isinstance(var, Variable) or isinstance(
                var, six.string_types
            ), "Wrong type for fetch_list[%s]: %s" % (i, type(var))
            global_block.append_op(
                type='fetch',
                inputs={'X': [var]},
                outputs={'Out': [fetch_var]},
                attrs={'col': i},
            )
    return tmp_program


class PrimNet(paddle.nn.Layer):
    def __init__(self, python_api):
        super(PrimNet, self).__init__()
        self.python_api = python_api

    def forward(self, x):
        out = self.python_api(x)
        return out


class PrimTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._np_rand_state = np.random.get_state()
        cls._py_rand_state = random.getstate()
        np.random.seed(123)
        random.seed(124)
        paddle.seed(2023)
        cls.enable_fw_comp = True
        cls.enable_rev_comp = True
        cls.enable_cinn = True
        cls.dtype = None
        cls.allowed_inputs_dtype = config.DTYPE

        cls.to_static_rtol = None
        cls.to_static_atol = None
        cls.fw_comp_rtol = None
        cls.fw_comp_atol = None
        cls.rev_comp_rtol = None
        cls.rev_comp_atol = None
        cls.cinn_rtol = None
        cls.cinn_atol = None

        cls.op_type = None  # comp or prim
        cls.api = None
        '''
        inputs(Sequence): numpy  of python api input
        '''
        cls.inputs = None
        cls.cotangents = None
        cls.grad_mask = None

        cls.eager_places = []  # []
        cls.static_places = []  # []
        cls.eager_api_out_desire = {}  # {}
        cls.eager_grad_desire = {}  # {}

        cls.init_custom_config()
        cls.data_prepare()

    @classmethod
    def init_custom_config(cls):
        # cls.np_inputs = None
        # cls.dtype = None
        # cls.op_type = None

        # cls.np_cotangents = None
        # cls.grad_mask = None
        pass

    @classmethod
    def python_api(cls, inputs):
        pass

    @classmethod
    def data_prepare(cls):
        if cls.__name__ is "PrimTest":
            return
        # pdb.set_trace()
        cls.check_op_type()
        cls.check_dtype()
        cls.check_custom_inputs()
        cls.check_grad_mask()
        cls.api = cls.python_api
        cls.check_python_api()
        cls.init_eager_places()
        cls.init_static_places()
        cls.inputs = convert_np_list_data_type(cls.inputs, cls.dtype)
        if cls.cotangents is not None:
            cls.cotangents = convert_np_list_data_type(
                _as_list(cls.cotangents), cls.dtype
            )
        cls.init_test_threshold()
        paddle.disable_static()
        cls.init_eager_desire()
        # pdb.set_trace()

    def setUp(self):
        core._set_prim_all_enabled(False)

    def tearDown(self):
        core._set_prim_all_enabled(False)

    @classmethod
    def tearDownClass(cls):
        np.random.set_state(cls._np_rand_state)
        random.setstate(cls._py_rand_state)

    @classmethod
    def set_grad_mask(cls, grad_mask):
        '''
        grad_mask(Sequence[bool]|Sequence[Sequence[bool]]): Specify which input needs gradient calculation,default to all.
        '''
        cls.grad_mask = grad_mask

    @classmethod
    def check_dtype(cls):
        if cls.dtype is None:
            raise TypeError("You must set dtype in your test.")
        if cls.dtype not in cls.allowed_inputs_dtype:
            raise TypeError(
                "The prim test framework does not support  %s." % cls.dtype
            )

    @classmethod
    def check_python_api(cls):
        if cls.api is None:
            raise NotImplementedError(
                "You must implement python_api function in your unittest class."
            )

    @classmethod
    def check_custom_inputs(cls):
        if cls.inputs is None:
            raise TypeError("You must init inputs with nparray")
        if not isinstance(cls.inputs, Sequence):
            raise TypeError("Inputs should be a sequence of nparray.")

    @classmethod
    def check_grad_mask(cls):
        if cls.grad_mask is None:
            return
        else:
            if not isinstance(cls.grad_mask, Sequence):
                raise TypeError("grad_mask should be a sequence.")
            assert len(cls.grad_mask) == len(cls.inputs), (
                f'len(grad_mask) shoule be equal to len(inputs), '
                f'but len(grad_mask)={len(cls.grad_mask)} and len(inputs)={len(cls.inputs)}.'
            )
            for i, input in enumerate(cls.inputs):
                assert type(cls.grad_mask[i]) == type(input), (
                    f'type(grad_mask{i}) should be the same as type(inputs{i}),'
                    f'but type(grad_mask{i})={type(cls.grad_mask[i])} and type(inputs{i})={type(cls.inputs[i])}.'
                )
                if isinstance(cls, Sequence):
                    assert len(cls.grad_mask[i]) == len(input), (
                        f'len(grad_mask{i}) should be the same as len(inputs{i}),'
                        f'but len(grad_mask{i})={len(cls.grad_mask[i])} and len(inputs{i})={len(cls.inputs[i])}.'
                    )

    @classmethod
    def check_op_type(cls):
        if cls.op_type is None:
            raise TypeError("You must set op_type in your test.")
        if cls.op_type is not "prim" and not "comp":
            raise TypeError("op_type must be one of 'prim' or 'comp'")

    @classmethod
    def init_eager_places(cls):
        # cls.eager_places.append("cpu")
        if paddle.is_compiled_with_cuda():
            cls.eager_places.append("gpu")
        print("init_eager_places", cls.eager_places, cls.__name__)

    @classmethod
    def init_static_places(cls):
        # cls.static_places.append(fluid.CPUPlace())
        if core.is_compiled_with_cuda():
            cls.static_places.append(fluid.CUDAPlace(0))

    @classmethod
    def init_test_threshold(cls):
        cls.to_static_rtol = (
            config.TOLERANCE.get(cls.dtype).get("to_static").get("rtol")
            if cls.to_static_rtol is None
            else cls.to_static_rtol
        )
        cls.to_static_atol = (
            config.TOLERANCE.get(cls.dtype).get("to_static").get("atol")
            if cls.to_static_atol is None
            else cls.to_static_atol
        )
        cls.fw_comp_rtol = (
            config.TOLERANCE.get(cls.dtype).get("fw_comp").get("rtol")
            if cls.fw_comp_rtol is None
            else cls.fw_comp_rtol
        )
        cls.fw_comp_atol = (
            config.TOLERANCE.get(cls.dtype).get("fw_comp").get("atol")
            if cls.fw_comp_atol is None
            else cls.fw_comp_atol
        )
        cls.rev_comp_rtol = (
            config.TOLERANCE.get(cls.dtype).get("rev_comp").get("rtol")
            if cls.rev_comp_rtol is None
            else cls.rev_comp_rtol
        )
        cls.rev_comp_atol = (
            config.TOLERANCE.get(cls.dtype).get("rev_comp").get("atol")
            if cls.rev_comp_atol is None
            else cls.rev_comp_atol
        )
        cls.cinn_rtol = (
            config.TOLERANCE.get(cls.dtype).get("cinn").get("rtol")
            if cls.cinn_rtol is None
            else cls.cinn_rtol
        )
        cls.cinn_atol = (
            config.TOLERANCE.get(cls.dtype).get("cinn").get("atol")
            if cls.cinn_atol is None
            else cls.cinn_atol
        )

    @classmethod
    def init_eager_desire(cls):
        for place in cls.eager_places:
            paddle.device.set_device(place)
            eager_inputs = gen_eager_data(cls.inputs, cls.dtype, place)
            eager_out = cls.api(eager_inputs)
            xs = get_need_grad_xs(eager_inputs, cls.grad_mask)
            vs = None
            if cls.cotangents is not None:
                vs = flatten(gen_eager_data(cls.cotangents, cls.dtype, place))
            dout = paddle.grad(
                eager_out, xs, grad_outputs=vs, allow_unused=True
            )
            if place is 'cpu':
                cls.eager_api_out_desire.update(
                    {str(fluid.CPUPlace()): eager_out}
                )
                cls.eager_grad_desire.update({str(fluid.CPUPlace()): dout})
            if place is 'gpu':
                cls.eager_api_out_desire.update(
                    {str(fluid.CUDAPlace(0)): eager_out}
                )
                cls.eager_grad_desire.update({str(fluid.CUDAPlace(0)): dout})
            cls.eager_api_out_desire.update({place: eager_out})
            cls.eager_grad_desire.update({place: dout})
        # print("eager_api_out_desire", cls.eager_api_out_desire)
        # print("eager_grad_desire", cls.eager_grad_desire)

    def get_eager_api_out_and_grad(
        self, api, np_inputs, np_vs, grad_mask, place
    ):
        print("api", api)
        eager_inputs = gen_eager_data(np_inputs, self.dtype, place)
        eager_out = api(eager_inputs)
        eager_out = _as_list(eager_out)
        print("eager_out", eager_out)
        xs = get_need_grad_xs(eager_inputs, grad_mask)
        vs = None
        if np_vs is not None:
            vs = flatten(gen_eager_data(np_vs, place))
        print("xs", xs)
        # print(api.forward.program_cache.last()[-1][-1].program)
        dxs = paddle.grad(eager_out, xs, grad_outputs=vs, allow_unused=True)
        print("dxs", dxs)
        return eager_out, dxs

    '''
    only for prim op
    test backward comp
    # '''

    @skip_base_class_test
    def test_eager_comp(self):
        if self.op_type is not "prim" or self.enable_rev_comp is False:
            return
        # print("self.inputs",self.inputs)
        # print("eager_api_out_desire", self.eager_api_out_desire)
        # print("eager_grad_desire", self.eager_grad_desire)
        paddle.disable_static()
        core._set_prim_backward_enabled(self.enable_rev_comp)
        for place in self.eager_places:
            paddle.device.set_device(place)
            _, eager_grad_actual = self.get_eager_api_out_and_grad(
                self.api, self.inputs, self.cotangents, self.grad_mask, place
            )
            if len(eager_grad_actual) != len(self.eager_grad_desire[place]):
                msg = (
                    "The output grad tensor's number of comp dygraph is different with dygraph on %s."
                    % (str(place))
                )
                raise RuntimeError(msg)

            for i in range(len(eager_grad_actual)):
                if not np.allclose(
                    eager_grad_actual[i],
                    self.eager_grad_desire[place][i],
                    rtol=self.rev_comp_rtol,
                    atol=self.rev_comp_atol,
                ):
                    msg = (
                        'Check eager comp grad failed. Mismatch between enable backward comp '
                        'and disable backward comp on %s, the output  grad tensor\'s index is : %d \n'
                        'enable backward comp:%s\n disable backward comp:%s\n'
                        % (
                            str(place),
                            i,
                            eager_grad_actual[i],
                            self.eager_grad_desire[place][i],
                        )
                    )
                    raise RuntimeError(msg)
        core._set_prim_backward_enabled(False)

    @skip_base_class_test
    def test_to_static(self):
        # print("self.inputs",self.inputs)
        # print("eager_api_out_desire", self.eager_api_out_desire)
        # print("eager_grad_desire", self.eager_grad_desire)
        paddle.disable_static()
        core._set_prim_all_enabled(False)
        for place in self.eager_places:
            paddle.device.set_device(place)
            net = PrimNet(self.api)
            net = apply_to_static(net, False)
            (
                to_static_out_actual,
                to_static_grad_actual,
            ) = self.get_eager_api_out_and_grad(
                net, self.inputs, self.cotangents, self.grad_mask, place
            )
            out_desire = flatten(_as_list(self.eager_api_out_desire[place]))
            # print("to_static_out_actual",to_static_out_actual)
            to_static_out_actual = flatten(_as_list(to_static_out_actual))
            # print("to_static_out_actual",to_static_out_actual)
            # print("out_desire", out_desire)
            # check to_static forward
            if len(to_static_out_actual) != len(out_desire):
                msg = (
                    "The to_static forward api out tensor nums is different with eager forward api out tensor nums on %s."
                    'to_static forward api out tensor nums = %s, eager forward api out tensor nums = %s. \n'
                    % (str(place), len(to_static_out_actual), len(out_desire))
                )
                raise RuntimeError(msg)

            for i in range(len(to_static_out_actual)):
                if not np.allclose(
                    to_static_out_actual[i].numpy(),
                    out_desire[i].numpy(),
                    rtol=self.to_static_rtol,
                    atol=self.to_static_atol,
                ):
                    msg = (
                        'Check to_static forward api out failed. Mismatch between to_static '
                        'and eager on %s, the forward api out tensor\'s index is : %d \n'
                        'to_static forward api out tensor:%s\n eager forward api out tensor:%s\n'
                        % (
                            str(place),
                            i,
                            to_static_out_actual[i],
                            out_desire[i],
                        )
                    )
                    raise RuntimeError(msg)
            # check to_static grad
            if len(to_static_grad_actual) != len(self.eager_grad_desire[place]):
                msg = (
                    "The to_static grad out tensor nums is different with eager fgrad out tensor nums on %s."
                    'to_static grad out tensor nums = %s, eager grad out tensor nums = %s. \n'
                    % (
                        str(place),
                        len(to_static_grad_actual),
                        len(self.eager_grad_desire[place]),
                    )
                )
                raise RuntimeError(msg)
            for i in range(len(to_static_grad_actual)):
                if not np.allclose(
                    to_static_grad_actual[i].numpy(),
                    self.eager_grad_desire[place][i].numpy(),
                    rtol=self.to_static_rtol,
                    atol=self.to_static_atol,
                ):
                    msg = (
                        'Check to_static grad failed. Mismatch between to_static '
                        'and eager on %s, the output  grad tensor\'s index is : %d \n'
                        'to_static grad: %s\n eager grad:%s\n'
                        % (
                            str(place),
                            i,
                            to_static_grad_actual[i].numpy(),
                            self.eager_grad_desire[place][i],
                        )
                    )
                    raise RuntimeError(msg)

    @skip_base_class_test
    def test_jit_comp(self):
        paddle.disable_static()
        if self.op_type is "prim":
            core._set_prim_backward_enabled(self.enable_rev_comp)
        else:
            core._set_prim_forward_enabled(self.enable_fw_comp)
            core._set_prim_backward_enabled(self.enable_rev_comp)
        # 组合测前反向，原子只需要测反向
        for place in self.eager_places:
            paddle.device.set_device(place)
            net = PrimNet(self.api)
            net = apply_to_static(
                net, core.is_compiled_with_cinn() and self.enable_cinn
            )
            jit_out_actual, jit_grad_actual = self.get_eager_api_out_and_grad(
                net, self.inputs, self.cotangents, self.grad_mask, place
            )
            # print(net.forward.program_cache.last()[-1][-1].program)
            # breakpoint()
            jit_out_actual = flatten(_as_list(jit_out_actual))
            out_desire, grad_out_desire = (
                flatten(_as_list(self.eager_api_out_desire[place])),
                self.eager_grad_desire[place],
            )
            forward_comp_status = (
                "on"
                if self.enable_fw_comp and self.op_type is 'comp'
                else "off"
            )
            backward_comp_status = "on" if self.enable_rev_comp else "off"
            cinn_status = (
                "on"
                if core.is_compiled_with_cinn() and self.enable_cinn
                else "off"
            )
            # check prim op jit forward comp
            if self.op_type is 'comp':
                if len(jit_out_actual) != len(out_desire):
                    msg = (
                        "The test_jit_comp test forward api out tensor nums is different with eager forward api out tensor nums on %s,"
                        ' when forward comp is %s, cinn is %s, jit forward api out tensor nums = %s, jit forward api out tensor nums = %s. \n'
                        % (
                            str(place),
                            forward_comp_status,
                            cinn_status,
                            len(jit_out_actual),
                            len(out_desire),
                        )
                    )
                    raise RuntimeError(msg)

                for i in range(len(jit_out_actual)):
                    if not np.allclose(
                        jit_out_actual[i].numpy(),
                        out_desire[i].numpy(),
                        rtol=self.cinn_rtol,
                        atol=self.cinn_rtol,
                    ):
                        msg = (
                            'Check test_jit_comp forward api out failed when forward comp is %s, cinn is %s.'
                            ' Mismatch between jit and eager on %s, the forward api out tensor\'s index is : %d \n'
                            'jit forward api out tensor:%s\n eager forward api out tensor:%s\n'
                            % (
                                forward_comp_status,
                                cinn_status,
                                str(place),
                                i,
                                jit_out_actual[i],
                                out_desire[i],
                            )
                        )
                        raise RuntimeError(msg)
            # check jit grad
            if len(jit_grad_actual) != len(grad_out_desire):
                msg = (
                    "The jit grad out tensor nums is different with eager grad out tensor nums on %s,"
                    'when forward comp is %s, backward comp is %s,cinn is %s,jit grad out tensor nums = %s, eager grad out tensor nums = %s. \n'
                    % (
                        str(place),
                        forward_comp_status,
                        backward_comp_status,
                        cinn_status,
                        len(jit_grad_actual),
                        len(grad_out_desire),
                    )
                )
                raise RuntimeError(msg)
            for i in range(len(jit_grad_actual)):
                if not np.allclose(
                    jit_grad_actual[i].numpy(),
                    grad_out_desire[i].numpy(),
                    rtol=self.cinn_rtol,
                    atol=self.cinn_atol,
                ):
                    msg = (
                        'Check test_jit_comp grad out failed when forward comp is %s, backward comp is %s,cinn is %s.'
                        ' Mismatch between jit and eager on %s, the jit grad out tensor\'s index is : %d \n'
                        'jit grad out tensor:%s\n eager forward api out tensor:%s\n'
                        % (
                            forward_comp_status,
                            backward_comp_status,
                            cinn_status,
                            str(place),
                            i,
                            jit_out_actual[i],
                            out_desire[i],
                        )
                    )
                    raise RuntimeError(msg)

    @skip_base_class_test
    def test_static_comp(self):
        paddle.enable_static()
        if self.op_type is "prim":
            core._set_prim_backward_enabled(self.enable_rev_comp)
        else:
            core._set_prim_forward_enabled(self.enable_fw_comp)
            core._set_prim_backward_enabled(self.enable_rev_comp)
        for place in self.static_places:
            if core.is_compiled_with_cinn() and self.enable_cinn:
                paddle.set_flags({'FLAGS_use_cinn': True})
            startup_program, main_program = (
                paddle.static.Program(),
                paddle.static.Program(),
            )
            with paddle.static.program_guard(main_program, startup_program):
                feed_dict = {}
                static_inputs, feed_inputs = gen_static_data_and_feed(
                    self.inputs, self.dtype, "x"
                )
                feed_dict.update(feed_inputs)
                vs = None
                if self.cotangents is not None:
                    vs, feed_vs = gen_static_data_and_feed(
                        self.cotangents, self.dtype, "v"
                    )
                    vs = flatten(vs)
                    feed_dict.update(feed_vs)
                static_out = self.api(static_inputs)
                static_out = flatten(_as_list(static_out))
                static_xs = get_need_grad_xs(static_inputs, self.grad_mask)
                static_dxs = paddle.static.gradients(static_out, static_xs, vs)
            exe = paddle.static.Executor(place)
            if core.is_compiled_with_cinn() and self.enable_cinn:
                main_program = _add_fetch_ops(
                    main_program, fetch_list=static_out + static_dxs
                )
                compiled_prog = paddle.static.CompiledProgram(
                    main_program
                ).with_data_parallel(loss_name=static_dxs[0].name)
            else:
                compiled_prog = main_program
            scope = paddle.static.Scope()
            with paddle.static.scope_guard(scope):
                exe.run(startup_program)
                print("main_program", main_program)
                print(feed_dict)
                exe_out = exe.run(
                    compiled_prog,
                    feed=feed_dict,
                    fetch_list=static_out + static_dxs,
                    return_numpy=True,
                )
            print("exe_out", exe_out)
            static_out_actual, static_grad_actual = (
                exe_out[0 : len(static_out)],
                exe_out[len(static_out) :],
            )
            print("static_out_actual", static_out_actual)
            print(self.eager_api_out_desire[str(place)])
            print("static_grad_actual", static_grad_actual)
            print(self.eager_grad_desire[str(place)])
