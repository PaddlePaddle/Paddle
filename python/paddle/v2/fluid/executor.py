import numpy as np
from . import core
from framework import Program, default_main_program, Parameter, Variable
import distribute_planner

__all__ = ['Executor', 'g_scope']

g_scope = core.Scope()


def as_numpy(tensor):
    if isinstance(tensor, list):
        return [as_numpy(t) for t in tensor]
    assert isinstance(tensor, core.LoDTensor)
    lod = tensor.lod()
    tensor_data = np.array(tensor)
    if len(lod) == 0:
        ans = tensor_data
    else:
        raise RuntimeError("LoD Calculate lacks unit tests and buggy")
    # elif len(lod) == 1:
    #     ans = []
    #     idx = 0
    #     while idx < len(lod) - 1:
    #         ans.append(tensor_data[lod[idx]:lod[idx + 1]])
    #         idx += 1
    # else:
    #     for l in reversed(lod):
    #         ans = []
    #         idx = 0
    #         while idx < len(l) - 1:
    #             ans.append(tensor_data[l[idx]:l[idx + 1]])
    #             idx += 1
    #         tensor_data = ans
    #     ans = tensor_data
    return ans


class Executor(object):
    def __init__(self, places):
        if not isinstance(places, list) and not isinstance(places, tuple):
            places = [places]

        act_places = []
        for each in places:
            p = core.Place()
            p.set_place(each)
            act_places.append(p)

        self.executor = core.Executor(act_places)
        self.places = places

    def optimize(self, optimize_ops, params_grads, program=None, **kwargs):
        """
            optimize the program for different runtime environment

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

        if kwargs.has_key("pservers"):
            return self._optimize_distributed(optimize_ops, program,
                                              params_grads, **kwargs)

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
        # remove optimize ops and add a send op to main_program
        # FIXME(typhoonzero): delete_op only remove the first accurence,
        # need to consider about multiple same optimize op?
        for op in optimize_ops:
            program.global_block().delete_op(op)
        if kwargs.has_key("split_method"):
            split_method = kwargs["split_method"]
        else:
            split_method = distribute_planner.round_robin

        assert (callable(split_method))
        pserver_endpoints = kwargs["pservers"].split(",")
        self.param_grad_map = split_method(params_and_grads, pserver_endpoints)

        for ep in pserver_endpoints:
            # FIXME(typhoonzero): send to different servers can run in parrallel.
            send_op = program.global_block().append_op(
                type="send",
                inputs={"X": self.param_grad_map[ep]["grads"]
                        },  # inputs is a list of tensors to be send
                outputs={},
                attrs={"endpoint": ep})

    def get_pserver_program(self, endpoint, optimize_ops):
        pserver_program = Program()
        for v in self.param_grad_map[endpoint]["params"]:
            self._clone_param(pserver_program.global_block(), v)

        optimize_sub_program = Program()
        for opt_op in optimize_ops:
            for varname, var in opt_op.inputs.iteritems():
                optimize_sub_program.global_block().create_var(
                    name=var.name,
                    persistable=var.persistable,
                    dtype=var.dtype,
                    shape=var.shape)
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
                "OptimizeProgram": optimize_sub_program.desc,
                "endpoint": endpoint,
                "ParamList":
                [p.name for p in self.param_grad_map[endpoint]["params"]],
                "GradList":
                [p.name for p in self.param_grad_map[endpoint]["grads"]]
            })
        pserver_program.sync_with_cpp()
        return pserver_program

    def aslodtensor(self, data):
        def accumulate(data):
            if not isinstance(data, list):
                return 1
            return sum([accumulate(sub) for sub in data])

        def parselod(data):
            seq_lens = [accumulate(seq) for seq in data]
            cur_len = 0
            lod = [cur_len]
            for l in seq_lens:
                cur_len += l
                lod.append(cur_len)
            return lod

        assert len(self.places) != 0
        if not isinstance(data, list):
            # pure tensor case
            tensor = core.LoDTensor()
            tensor.set(data, self.places[0])
            return tensor
        else:
            raise RuntimeError("Current implementation lacks unittests")
            # lodtensor case
            lod = []
            if not isinstance(data[0], list):
                lod.append(parselod(data))
                flattened_data = np.concatenate(data, axis=0).astype("int64")
            else:
                while isinstance(data[0], list):
                    lod.append(parselod(seq))
                    flattened_data = [item for seq in data for item in seq]
                    data = flattened_data
                flattened_data = np.concatenate(data, axis=0).astype("int64")
            flattened_data = flattened_data.reshape([len(flattened_data), 1])
            tensor = core.LoDTensor()
            tensor.set(flattened_data, self.places[0])
            tensor.set_lod(lod)
            return tensor

    def run(self,
            program=None,
            feed=None,
            fetch_list=None,
            feed_var_name='feed',
            fetch_var_name='fetch',
            scope=None,
            return_numpy=True):
        if feed is None:
            feed = {}
        if fetch_list is None:
            fetch_list = []

        if program is None:
            program = default_main_program()

        if not isinstance(program, Program):
            raise TypeError()

        if scope is None:
            scope = g_scope

        program = program.clone()
        global_block = program.global_block()
        feed_var = global_block.create_var(
            name=feed_var_name,
            type=core.VarDesc.VarType.FEED_MINIBATCH,
            persistable=True)

        for i, name in enumerate(feed):
            out = global_block.var(name)
            global_block.prepend_op(
                'feed',
                inputs={'X': [feed_var]},
                outputs={'Out': [out]},
                attrs={'col': i})
            cur_feed = feed[name]
            if not isinstance(cur_feed, core.LoDTensor):
                cur_feed = self.aslodtensor(cur_feed)
            core.set_feed_variable(scope, cur_feed, feed_var.name, i)

        fetch_var = global_block.create_var(
            name=fetch_var_name,
            type=core.VarDesc.VarType.FETCH_LIST,
            persistable=True)
        for i, var in enumerate(fetch_list):
            global_block.append_op(
                type='fetch',
                inputs={'X': [var]},
                outputs={'Out': [fetch_var]},
                attrs={'col': i})

        self.executor.run(program.desc, scope, 0, True, True)
        outs = [
            core.get_fetch_variable(scope, fetch_var_name, i)
            for i in xrange(len(fetch_list))
        ]

        if return_numpy:
            outs = as_numpy(outs)
        return outs
