from __future__ import print_function
import framework
from framework import Program, default_main_program, Parameter, Variable
import optimizer
from layer_helper import LayerHelper
from distributed_spliter import *
import math


class VarBlock:
    def __init__(self, varname, offset, size):
        self.varname = varname
        # NOTE: real offset is offset * size
        self.offset = offset
        self.size = size

    def __str__(self):
        return "%s:%d:%d" % (self.varname, self.offset, self.size)


def split_dense_variable(var_list,
                         pserver_count,
                         min_block_size=1024,
                         max_block_size=1048576):
    """
        We may need to split dense tensor to one or several blocks and put
        them equally onto parameter server. One block is a sub-tensor
        aligned by dim[0] of the tensor.
        
        We need to have a minimal block size so that the calculations in
        the parameter server side can gain better performance. By default
        mininum block size is 1024. The max block size is used to prevent
        too large block that may causing send error.
    """
    blocks = []
    for var in var_list:
        split_count = pserver_count
        var_numel = reduce(lambda x, y: x * y, var.shape)
        max_pserver_count = int(math.floor(var_numel / float(min_block_size)))
        if max_pserver_count == 0:
            max_pserver_count = 1
        if max_pserver_count < pserver_count:
            split_count = max_pserver_count
        block_size = int(math.ceil(var_numel / float(split_count)))

        if len(var.shape) >= 2:
            # align by dim1(width)
            dim1 = reduce(lambda x, y: x * y, var.shape[1:])
            remains = block_size % dim1
            if remains != 0:
                block_size += dim1 - remains
        # update split_count after align
        split_count = int(math.ceil(var_numel / float(block_size)))
        for block_id in xrange(split_count):
            curr_block_size = min(block_size, var_numel - (
                (block_id) * block_size))
            block = VarBlock(var.name, block_id, curr_block_size)
            blocks.append(str(block))
        print("$$ splited var: ", var.name, var.shape, split_count, len(blocks),
              block_size)
    return blocks


class DistributeTranspiler:
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

            :param optimize_ops: op list of optimization, should be the
                                 return value of Optimizer.minimize
            :type optimize_ops: list
            :param program: program to optimize, default default_main_program
            :param pservers: parameter server endpoints like "m1:6174,m2:6174"
            :type pservers: string
            :return: return a list of programs
        """
        assert (callable(split_method))
        if program is None:
            program = default_main_program()
        self.program = program
        self.trainers = trainers
        self.optimize_ops = optimize_ops
        # steps to transpile:
        # 1. split variable to multiple blocks, align by product(dim[1:]) (width).
        # 2. modify trainer program add split_op to each Grad.
        # 3. append send_op to trainer.
        # 4. append concat_op to trainer to update local weights.
        # 5. create new program as parameter server.
        # 5. create parameter server program by split_method generated endpoint->VarBlock
        # 6. run compile time infershape for parameter server program

        pserver_endpoints = pservers.split(",")

        # step1
        param_list = [pg[0] for pg in params_grads]
        grad_list = [pg[1] for pg in params_grads]
        # TODO: add split selected rows support
        grad_blocks = split_dense_variable(grad_list, len(pserver_endpoints))
        param_blocks = split_dense_variable(param_list, len(pserver_endpoints))
        # step2
        grad_var_mapping = self._append_split_op(program, grad_blocks)

        # step3
        send_inputs = []
        send_outputs = []
        for _, splited in grad_var_mapping.iteritems():
            send_inputs.extend(splited)
        param_var_mapping = self._create_vars_from_blocklist(program,
                                                             param_blocks)
        for _, splited in param_var_mapping.iteritems():
            send_outputs.extend(splited)
        # let send_op know which endpoint to send which var, eplist is of the same
        # order of send_inputs.
        eplist = split_method(send_inputs, pserver_endpoints)

        send_op = program.global_block().append_op(
            type="send",
            inputs={"X": send_inputs},
            outputs={"Out": send_outputs},
            attrs={"endpoints": pserver_endpoints,
                   "epmap": eplist})

        # step4
        for varname, splited_var in param_var_mapping.iteritems():
            if len(splited_var) <= 1:
                continue
            orig_param = program.global_block().vars[varname]
            concat = program.global_block().append_op(
                type="concat",
                inputs={"X": splited_var},
                outputs={"Out": orig_param},
                attrs={"axis": 0})

    def _create_vars_from_blocklist(self, program, block_list):
        block_map = dict()
        var_mapping = dict()
        for block_str in block_list:
            varname, offset, size = block_str.split(":")
            if not block_map.has_key(varname):
                block_map[varname] = []
            block_map[varname].append((long(offset), long(size)))
        for varname, splited in block_map.iteritems():
            orig_var = program.global_block().vars[varname]
            var_mapping[varname] = []
            if len(splited) == 1:
                var_mapping[varname] = [orig_var]
                continue
            orig_shape = orig_var.shape
            orig_dim1_flatten = 1
            if len(orig_shape) >= 2:
                orig_dim1_flatten = reduce(lambda x, y: x * y, orig_shape[1:])

            for i, block in enumerate(splited):
                size = block[1]
                rows = size / orig_dim1_flatten
                splited_shape = [rows]
                if len(orig_shape) >= 2:
                    splited_shape.extend(orig_shape[1:])
                var = program.global_block().create_var(
                    name="%s.block%d" % (varname, i),
                    psersistable=False,
                    dtype=orig_var.dtype,
                    shape=splited_shape)  # flattend splited var
                var_mapping[varname].append(var)
        return var_mapping

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

    def _append_split_op(self, program, gradblocks):
        var_mapping = self._create_vars_from_blocklist(program, gradblocks)
        for varname, splited_vars in var_mapping.iteritems():
            # variable that don't need to split have empty splited_vars
            if len(splited_vars) <= 1:
                continue
            orig_var = program.global_block().vars[varname]
            sections = []
            for v in splited_vars:
                sections.append(v.shape[0])
            program.global_block().append_op(
                type="split",
                inputs={"X": orig_var},
                outputs={"Out": splited_vars},
                attrs={"sections": sections}  # assume split evenly
            )
        return var_mapping

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
                "OptimizeProgram": optimize_sub_program.desc,
                "endpoint": endpoint,
                "ParamList":
                [p.name for p in self.param_grad_map[endpoint]["params"]],
                "GradList":
                [p.name for p in self.param_grad_map[endpoint]["grads"]],
                "Trainers": self.trainers
            })
        pserver_program.sync_with_cpp()
        return pserver_program
