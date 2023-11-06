#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import contextlib
import unittest

import numpy as np

import paddle
from paddle import _legacy_C_ops, base
from paddle.base import core, framework
from paddle.base.dygraph.base import switch_to_static_graph

paddle.enable_static()


@contextlib.contextmanager
def program_scope_guard():
    prog = base.Program()
    startup_prog = base.Program()
    scope = base.core.Scope()
    with base.scope_guard(scope):
        with base.program_guard(prog, startup_prog):
            with base.unique_name.guard():
                yield


@switch_to_static_graph
def _add_build_strategy_for(input_program, start_op_index, end_op_index):
    compiled_program = paddle.static.CompiledProgram(
        core.Graph(input_program.desc, start_op_index, end_op_index),
        build_strategy=paddle.static.BuildStrategy(),
    )
    compiled_program._compile(
        core.Scope(), paddle.framework._current_expected_place()
    )
    ir_graph = paddle.base.framework.IrGraph(compiled_program._graph)
    builded_program = ir_graph.to_program()
    return builded_program


@switch_to_static_graph
def _build_program_by_desc(program_desc):
    prog = framework.Program()
    prog.desc = program_desc
    prog.blocks = [
        framework.Block(prog, i) for i in range(prog.desc.num_blocks())
    ]
    prog._sync_with_cpp()
    return prog


# NOTE: Because RunProgramOp has a special output of type std::vector<Scope *>,
# the OpTest cannot be used in RunProgramOp. The variable type cannot be specified
# when creating output variables in OpTest, default type is LoDTensor
# NOTE: the gradient test method in OpTest also cannot be used for RunProgramOp,
# because it hold BlockDesc type attr, OperatorFactory can't parse this attr type
# when create Operator, so here compare gradients with static graph
# NOTE: Here rewrite a simple unittest framework for RunProgramOp
class RunProgramOpTest(unittest.TestCase):
    def build_model(self):
        raise NotImplementedError(
            "RunProgramOp test should implement build_model"
        )

    def check_output(self):
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for place in places:
            # TODO: RunProgramOp is not recommended for use in static graph mode now
            self.expect_outs = self.run_static_model(place, is_test=True)
            self.check_output_with_place(place)

    def check_grad(self):
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for place in places:
            # TODO: RunProgramOp is not recommended for use in static graph mode now
            self.expect_grads = self.run_static_model(place, is_test=False)
            self.check_grad_with_place(place)

    def run_static_model(self, place, is_test=True):
        with program_scope_guard():
            startup_program = base.default_startup_program()
            main_program = base.default_main_program()

            self.build_model()

            exe = base.Executor(place)
            exe.run(startup_program)

            if is_test:
                fetch_list = self.output_names['Out']
            else:
                fetch_list = self.get_param_grad_names()

            outs = exe.run(
                main_program, feed=self.inputs['X'], fetch_list=fetch_list
            )
            return outs

    def get_program_desc(self):
        with program_scope_guard():
            fwd_op_num = self.build_model()
            return base.default_main_program().desc, fwd_op_num

    def get_forward_backward_program_desc(
        self, whole_program_desc, forward_op_num, output_num
    ):
        program = _build_program_by_desc(whole_program_desc)
        forward_program = _add_build_strategy_for(program, 0, forward_op_num)
        backward_program = _add_build_strategy_for(
            program,
            forward_op_num + output_num,
            program.desc.block(0).op_size(),
        )
        return forward_program.desc, backward_program.desc

    def prepare_attrs(self):
        return [
            'global_block',
            self.program_desc.block(0),
            'start_op_index',
            0,
            'end_op_index',
            self.fwd_op_num,
            'program_id',
            paddle.utils._hash_with_id(self.program_desc, self),
        ]

    def get_param_grad_names(self):
        grad_names = []
        for var_name in self.inputs['Params']:
            grad_names.append(var_name + core.grad_var_suffix())
        return grad_names

    def check_output_with_place(self, place):
        # Step 1. run op
        actual_outs = self.calc_dygraph_output(place)

        # Step 2. compare output
        for expect_v, actual_v in zip(self.expect_outs, actual_outs):
            np.testing.assert_allclose(
                expect_v, actual_v.numpy(), rtol=1e-05, atol=1e-05
            )

    def check_grad_with_place(self, place):
        # Step 1. calc grads
        actual_grads = self.calc_dygraph_grad(place)

        # Step 2. compare grads
        for expect_v, actual_v in zip(self.expect_grads, actual_grads):
            np.testing.assert_array_almost_equal(expect_v, actual_v)
            np.testing.assert_allclose(
                expect_v, actual_v, rtol=1e-05, atol=1e-05
            )

    def prepare_dygraph_input(self, place, return_param_list=False):
        def create_var_base(is_input, name, np_value, stop_gradient):
            var = core.eager.Tensor(
                value=np_value, name=name, place=place, zero_copy=True
            )
            var.stop_gradient = stop_gradient
            return var

        # build inputs
        inputs = {}
        param_list = []
        inputs['X'] = []
        for name, np_value in self.inputs['X'].items():
            var = create_var_base(True, name, np_value, True)
            inputs['X'].append(var)
        inputs['Params'] = []
        for name, np_value in self.inputs['Params'].items():
            var = create_var_base(True, name, np_value, False)
            inputs['Params'].append(var)
            if return_param_list:
                param_list.append(var)

        if return_param_list:
            return inputs, param_list
        return inputs

    def prepare_dygraph_output(self):
        def create_var_base(is_input, name):
            var = framework._create_tensor(dtype=None, shape=None, name=name)
            var.stop_gradient = False
            return var

        # build outputs
        outputs = {}
        outputs['Out'] = []
        for name in self.output_names['Out']:
            outputs['Out'].append(create_var_base(False, name))

        outputs['OutScope'] = [core.Scope()]

        return outputs

    def calc_dygraph_output(self, place):
        self.program_desc, self.fwd_op_num = self.get_program_desc()
        self.attrs = self.prepare_attrs()

        with base.dygraph.guard(place):
            inputs = self.prepare_dygraph_input(place)
            outputs = self.prepare_dygraph_output()

            (
                forward_program_desc,
                backward_program_desc,
            ) = self.get_forward_backward_program_desc(
                self.program_desc, self.fwd_op_num, len(outputs['Out'])
            )

            use_interpretorcore = True
            self.attrs.extend(('use_interpretorcore', use_interpretorcore))
            if use_interpretorcore:
                self.attrs.extend(
                    (
                        'forward_global_block',
                        forward_program_desc.block(0),
                        'backward_global_block',
                        backward_program_desc.block(0),
                    )
                )

            self.attrs.extend(
                (
                    'param_grad_names',
                    [p.name + '@GRAD' for p in inputs['Params']],
                    'out_grad_names',
                    [out.name + '@GRAD' for out in outputs['Out']],
                    'x_grad_names',
                    [p.name + '@GRAD' for p in inputs['X']],
                    'x_names',
                    [t.name for t in inputs['X']],
                )
            )

            _legacy_C_ops.run_program(
                inputs['X'],
                inputs['Params'],
                outputs['Out'],
                outputs['OutScope'],
                None,
                *self.attrs
            )

            return outputs['Out']

    def calc_dygraph_grad(self, place):
        self.program_desc, self.fwd_op_num = self.get_program_desc()
        self.attrs = self.prepare_attrs()

        with base.dygraph.guard(place):
            # Step 1. run forward
            inputs, input_param_list = self.prepare_dygraph_input(place, True)
            outputs = self.prepare_dygraph_output()

            (
                forward_program_desc,
                backward_program_desc,
            ) = self.get_forward_backward_program_desc(
                self.program_desc, self.fwd_op_num, len(outputs['Out'])
            )

            use_interpretorcore = True
            self.attrs.extend(('use_interpretorcore', use_interpretorcore))
            if use_interpretorcore:
                self.attrs.extend(
                    (
                        'forward_global_block',
                        forward_program_desc.block(0),
                        'backward_global_block',
                        backward_program_desc.block(0),
                    )
                )

            self.attrs.extend(
                (
                    'param_grad_names',
                    [p.name + '@GRAD' for p in inputs['Params']],
                    'out_grad_names',
                    [out.name + '@GRAD' for out in outputs['Out']],
                    'x_grad_names',
                    [p.name + '@GRAD' for p in inputs['X']],
                    'x_names',
                    [t.name for t in inputs['X']],
                )
            )

            _legacy_C_ops.run_program(
                inputs['X'],
                inputs['Params'],
                outputs['Out'],
                outputs['OutScope'],
                None,
                *self.attrs
            )

            for param in input_param_list:
                var_type = self._get_grad_vartype(param.name)
                if var_type is None:
                    continue
                param._set_grad_type(var_type)

            # Step 2. run backward
            # NOTE: in unittest, only support single output now
            actual_outs = outputs['Out']
            assert len(actual_outs) == 1
            actual_outs[0].backward()

            # Step 3. prepare grads
            grads = []
            for param in input_param_list:
                grad = param.gradient()
                grads.append(grad)
            return grads

    def _get_grad_vartype(self, name):
        assert self.program_desc is not None
        grad_name = name + core.grad_var_suffix()
        for i in range(self.program_desc.num_blocks()):
            block = self.program_desc.block(i)
            var_desc = block.find_var_recursive(grad_name.encode())
            return var_desc.type() if var_desc is not None else None


class TestRunProgramOpWithFC(RunProgramOpTest):
    def setUp(self):
        self.op_type = "run_program"
        self.dtype = np.float32
        self.input_names = {
            'X': ['img'],
            'Params': ['weight_param', 'bias_param'],
        }
        self.output_names = {'Out': ['fc_0.tmp_2']}

        self.inputs = {
            'X': {
                self.input_names['X'][0]: np.random.random(
                    (32, 1, 28, 28)
                ).astype(self.dtype)
            },
            'Params': {
                self.input_names['Params'][0]: np.random.random(
                    (784, 10)
                ).astype(self.dtype),
                self.input_names['Params'][1]: np.random.random(
                    (32, 10)
                ).astype(self.dtype),
            },
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad()

    def build_model(self):
        # 1. simple model
        img = paddle.static.data(
            name=self.input_names['X'][0],
            shape=[None, 1, 28, 28],
            dtype='float32',
        )
        weight_attr = base.ParamAttr(
            name=self.input_names['Params'][0],
            learning_rate=0.5,
            initializer=paddle.nn.initializer.Assign(
                self.inputs['Params'][self.input_names['Params'][0]]
            ),
            trainable=True,
        )
        bias_attr = base.ParamAttr(
            name=self.input_names['Params'][1],
            learning_rate=0.5,
            initializer=paddle.nn.initializer.Assign(
                self.inputs['Params'][self.input_names['Params'][1]]
            ),
            trainable=True,
        )
        pred = paddle.static.nn.fc(
            x=img,
            size=10,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            activation='relu',
        )
        # 2. get forward op num
        fwd_op_num = base.default_main_program().global_block().desc.op_size()
        # 3. append backward
        grads = base.backward.gradients(targets=[pred], inputs=[img])

        return fwd_op_num


class TestRunProgramOpWithEmbedding(RunProgramOpTest):
    def setUp(self):
        self.op_type = "run_program"
        self.dtype = np.float32
        self.input_names = {'X': ['x'], 'Params': ['emb_weight']}
        self.output_names = {'Out': ['sum_0.tmp_0']}

        self.inputs = {
            'X': {'x': np.array([[1, 3, 0, 4, 7]]).astype("int64")},
            'Params': {
                'emb_weight': np.random.random(size=(10, 16)).astype("float32")
            },
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        # NOTE: fecth not support SelectedRows, catnot compare
        # sparse gradients with staic mode, only run dygraph
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for place in places:
            # TODO: RunProgramOp is not recommended for use in static graph mode now
            self.calc_dygraph_grad(place)

    def build_model(self):
        # 1. simple model
        x = paddle.static.data(
            name=self.input_names['X'][0], shape=[-1, 5], dtype='int64'
        )
        emb = paddle.static.nn.embedding(
            input=x,
            size=[10, 16],
            param_attr=base.ParamAttr(
                name="emb_weight",
                learning_rate=10,
                initializer=paddle.nn.initializer.Assign(
                    self.inputs['Params'][self.input_names['Params'][0]]
                ),
            ),
            is_sparse=True,
        )
        y = paddle.sum(emb, axis=-1)
        # 2. get forward op num
        fwd_op_num = base.default_main_program().global_block().desc.op_size()
        # 3. append backward
        grads = base.backward.gradients(targets=[y], inputs=[x])

        return fwd_op_num


class Net(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.fc1 = paddle.nn.Linear(10, 10)
        self.fc2 = paddle.nn.Linear(10, 1)

    def forward(self, x):
        out = self.fc1(x)
        out.stop_gradient = True
        out = self.fc2(out)
        return out


class TestParametersWithStopGradient(unittest.TestCase):
    def setUp(self):
        self.seed = 2021
        self.iter = 5

    def train(self, to_static):
        # prepare env
        paddle.seed(self.seed)

        net = Net()
        if to_static:
            net = paddle.jit.to_static(net)
        sgd = paddle.optimizer.SGD(0.01, parameters=net.parameters())

        for i in range(self.iter):
            x = paddle.rand([4, 10])
            out = net(x)
            loss = paddle.mean(out)

            loss.backward()
            sgd.minimize(loss)
            net.clear_gradients()

        return loss

    def test_stop_gradient(self):
        paddle.disable_static()

        dy_loss = self.train(to_static=False)
        st_loss = self.train(to_static=True)
        self.assertEqual(dy_loss, st_loss)

        paddle.enable_static()


if __name__ == "__main__":
    unittest.main()
