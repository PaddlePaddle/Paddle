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

from ..framework import Program, default_main_program, Parameter, Variable
from ..layer_helper import LayerHelper


def hash_name_to_server(params_grads, pserver_endpoints):
    """
    :param param_grads:
    :return: a map of pserver endpoint -> 
                    params -> [param list]
                    grads  -> [grad list]
    """

    def _hash_param(param_name, total):
        return hash(param_name) % total

    param_grad_map = dict()
    for param, grad in params_grads:
        if param.trainable is True and grad is not None:
            server_id = _hash_param(param.name, len(pserver_endpoints))
            server_for_param = pserver_endpoints[server_id]
            if not param_grad_map.has_key(server_for_param):
                param_grad_map[server_for_param] = {"params": [], "grads": []}
            param_grad_map[server_for_param]["params"].append(param)
            param_grad_map[server_for_param]["grads"].append(grad)

    return param_grad_map


def round_robin(params_grads, pserver_endpoints):
    assert (len(params_grads) > len(pserver_endpoints))

    param_grad_map = dict()
    pserver_idx = 0
    for param, grad in params_grads:
        if param.trainable is True:
            server_for_param = pserver_endpoints[pserver_idx]
            if not param_grad_map.has_key(server_for_param):
                param_grad_map[server_for_param] = {"params": [], "grads": []}

            param_grad_map[server_for_param]["params"].append(param)
            param_grad_map[server_for_param]["grads"].append(grad)

            pserver_idx += 1
            if pserver_idx >= len(pserver_endpoints):
                pserver_idx = 0
    return param_grad_map


class SimpleDistributeTranspiler:
    def transpile(self,
                  optimize_ops,
                  params_grads,
                  program=None,
                  pservers="127.0.0.1:6174",
                  trainers=1,
                  split_method=round_robin):
        """
            Transpile the program to a distributed data-parallelism programs.

            The main_program will be transform to use a remote parameter server
            to do parameter optimization. And the optimization graph will be put
            in to a parameter server program.

            Use different methods to split trainable varialbles to different
            parameter servers.

            Example to run:

            exe = fluid.Executor(place)
            t = fluid.DistributeTranspiler()
            t.transpile(optimize_ops, params_grads, pservers="127.0.0.1:6174", trainers=1)

            pserver_endpoint = os.getenv("PSERVER")
            if pserver_endpoint:
                pserver_prog = t.get_pserver_program(pserver_endpoint, optimize_ops)
                exe.run(fluid.default_startup_program())
                exe.run(pserver_prog)
            else:
                feeder = fluid.DataFeeder(feed_list=[images, label], place=place)
                exe.run(fluid.default_startup_program())

                for pass_id in range(PASS_NUM):
                    ...

            :param optimize_ops: op list of optimization, should be the
                                 return value of Optimizer.minimize
            :type optimize_ops: list
            :param program: program to optimize, default default_main_program
            :param pservers: parameter server endpoints like "m1:6174,m2:6174"
            :type pservers: string

            :return: return a list of programs
        """
        if program is None:
            program = default_main_program()
        self.program = program
        self.trainers = trainers
        self.optimize_ops = optimize_ops
        self._optimize_distributed(
            optimize_ops,
            program,
            params_grads,
            pservers=pservers,
            trainers=trainers,
            split_method=split_method)

    def _clone_param(self, block, v):
        assert isinstance(v, Parameter)
        new_p = Parameter(
            block=block,
            shape=v.shape,
            dtype=v.dtype,
            type=v.type,
            lod_level=v.lod_level,
            stop_gradient=v.stop_gradient,
            trainable=v.trainable,
            optimize_attr=v.optimize_attr,
            regularizer=v.regularizer,
            name=v.name)
        block.vars[new_p.name] = new_p

    def _clone_var(self, block, var):
        assert isinstance(var, Variable)
        return block.create_var(
            name=var.name,
            shape=var.shape,
            dtype=var.dtype,
            type=var.type,
            lod_level=var.lod_level,
            persistable=var.persistable)

    def _optimize_distributed(self, optimize_ops, program, params_and_grads,
                              **kwargs):
        if kwargs.has_key("split_method"):
            split_method = kwargs["split_method"]
        else:
            split_method = round_robin

        assert (callable(split_method))
        pserver_endpoints = kwargs["pservers"].split(",")
        self.param_grad_map = split_method(params_and_grads, pserver_endpoints)

        send_op_ordered_inputs = []
        send_op_ordered_outputs = []
        epmap = []
        for ep, v in self.param_grad_map.iteritems():
            send_op_ordered_inputs.extend(v["grads"])
            send_op_ordered_outputs.extend(v["params"])
            for i in v["grads"]:
                epmap.append(ep)
        send_op = program.global_block().append_op(
            type="send",
            inputs={"X": send_op_ordered_inputs
                    },  # inputs is a list of tensors to be send
            outputs={"Out": send_op_ordered_outputs},
            attrs={"endpoints": pserver_endpoints,
                   "epmap": epmap})

    def get_trainer_program(self):
        # remove optimize ops and add a send op to main_program
        self.program.global_block().delete_ops(self.optimize_ops)
        return self.program

    def _create_var_for_trainers(self, block, var, trainers):
        var_list = []
        for i in xrange(trainers):
            var_each = block.create_var(
                name="%s.trainer_%d" % (var.name, i),
                psersistable=var.persistable,
                dtype=var.dtype,
                shape=var.shape)
            var_list.append(var_each)
        return var_list

    def get_pserver_program(self, endpoint, optimize_ops):
        pserver_program = Program()
        for v in self.param_grad_map[endpoint]["params"]:
            self._clone_param(pserver_program.global_block(), v)

        optimize_sub_program = Program()
        grad_var_names = [
            var.name for var in self.param_grad_map[endpoint]["grads"]
        ]
        for opt_op in optimize_ops:
            for _, var in opt_op.inputs.iteritems():
                # NOTE: append operators to merge gradients from multiple
                # trainers. If trainers == 1, this is not needed.
                if self.trainers > 1 and var.name in grad_var_names:
                    vars2merge = self._create_var_for_trainers(
                        optimize_sub_program.global_block(), var, self.trainers)
                    merged_var = optimize_sub_program.global_block().create_var(
                        name=var.name,
                        persistable=var.persistable,
                        dtype=var.dtype,
                        shape=var.shape)
                    optimize_sub_program.global_block().append_op(
                        type="sum",
                        inputs={"X": vars2merge},
                        outputs={"Out": merged_var})
                    optimize_sub_program.global_block().append_op(
                        type="scale",
                        inputs={"X": merged_var},
                        outputs={"Out": merged_var},
                        attrs={"scale": 1.0 / float(self.trainers)})
                else:
                    optimize_sub_program.global_block().create_var(
                        name=var.name,
                        persistable=var.persistable,
                        dtype=var.dtype,
                        shape=var.shape)

            if opt_op.inputs.has_key("Grad"):
                if opt_op.inputs["Grad"].name in grad_var_names:
                    optimize_sub_program.global_block().append_op(
                        type=opt_op.type,
                        inputs=opt_op.inputs,
                        outputs=opt_op.outputs,
                        attrs=opt_op.attrs)
            else:
                optimize_sub_program.global_block().append_op(
                    type=opt_op.type,
                    inputs=opt_op.inputs,
                    outputs=opt_op.outputs,
                    attrs=opt_op.attrs)
        pserver_program.global_block().append_op(
            type="recv",
            inputs={"RX":
                    self.param_grad_map[endpoint]["grads"]},  # grads to recv
            outputs={},
            attrs={
                "OptimizeBlock": optimize_sub_program.global_block(),
                "endpoint": endpoint,
                "ParamList":
                [p.name for p in self.param_grad_map[endpoint]["params"]],
                "GradList":
                [p.name for p in self.param_grad_map[endpoint]["grads"]],
                "Trainers": self.trainers
            })
        pserver_program.sync_with_cpp()
        return pserver_program
