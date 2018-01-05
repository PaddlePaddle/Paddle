from __future__ import print_function
import framework
from framework import Program, default_main_program, Parameter, Variable
import optimizer
from layer_helper import LayerHelper
from distributed_spliter import *


class VarBlock:
    def __init__(self, varname, offset, size):
        self.varname = varname
        # NOTE: real offset is offset * size
        self.offset = offset
        self.size = size

    def __str__(self):
        return "%s:%d:%d" % (self.varname, self.offset, self.size)


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

        if kwargs.has_key("split_method"):
            split_method = kwargs["split_method"]
        else:
            split_method = round_robin
        pserver_endpoints = kwargs["pservers"].split(",")

        grad2param = dict()
        for param, grad in params_and_grads:
            grad2param[grad.name()] = param.name()

        # step1
        param_list = [pg[0] for pg in params_and_grads]
        grad_list = [pg[1] for pg in params_and_grads]
        # TODO: add split selected rows support
        grad_blocks = _split_dense_variable(grad_list, len(pserver_endpoints))
        param_blocks = _split_dense_variable(param_list, len(pserver_endpoints))
        ep2gradblock = split_method(grad_blocks, pserver_endpoints)
        # self.param_grad_map
        # step2
        var2splited = self._split_trainer_vars(program, grad_blocks)

        # step3
        send_inputs = []
        send_outputs = []
        for _, splited in var2splited.iteritems():
            send_inputs.extend(splited)
        send_outputs = self._create_vars_from_blocklist(program, param_blocks)

        send_op = program.global_block().append_op(
            type="send",
            inputs={"X": send_inputs},
            outputs={"Out": send_outputs},
            attrs={"endpoints": pserver_endpoints,
                   "epmap": epmap})

    def _create_vars_from_blocklist(self, program, block_list):
        block_map = dict()
        ret_vars = []
        for block_str in block_list:
            varname, offset, size = block_str.split(":")
            if not block_map.has_key(varname):
                block_map[varname] = []
            block_map[varname].append((long(offset), long(size)))

        for varname, splited in block_map.iteritems():
            orig_var = program.global_block().vars[varname]
            for block in splited:
                size = block[1]
                var = program.global_block().create_var(
                    name="%s.block%d" % (varname, i),
                    psersistable=False,
                    dtype=orig_var.dtype,
                    shape=[1, size])  # flattend splited var
                ret_vars.append(var)
        return ret_vars

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

    def _split_dense_variable(self,
                              var_list,
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
        block_sizes = []
        blocks = []
        for grad in var_list:
            dim1 = reduce(lambda x, y: x * y, grad.shape[1:])
            grad_numel = reduce(lambda x, y: x * y, grad.shape)
            if grad_numel < min_block_size:
                block_sizes.append(grad_numel)
            block_size = grad_numel / min_block_size
            if block_size < min_block_size:
                block_size = min_block_size
            # align by dim1(width)
            remains = block_size % dim1
            if remains != 0:
                block_size += dim1 - remains
            block_sizes.append(block_size)
            num_blocks = grad_numel / block_size
            print("grad numel :%d, blocksize: %d" % grad_numel, block_size)
            for block_id in xrange(num_blocks):
                block = VarBlock(grad.name(), block_id, block_size)
                blocks.append(str(block))
        return blocks

    def _split_trainer_vars(self, program, gradblocks, params_and_grads):
        var2blocks = dict()
        splited = dict()
        for block_str in gradblocks:
            varname, offset, size = block_str.split(":")
            if not var2blocks.has_key(varname):
                var2blocks[varname] = []
            var2blocks[varname].append((long(offset), long(size)))
        for varname, blocks in var2blocks.iteritems():
            orig_var = program.global_block().vars[varname]
            split_outs = []
            for i in xrange(len(blocks)):
                size = blocks[i][1]
                var = program.global_block().create_var(
                    name="%s.block%d" % (varname, i),
                    psersistable=False,
                    dtype=orig_var.dtype,
                    shape=[1, size])  # flattend splited var
                split_outs.append(var)

            splited[varname] = split_outs
            program.global_block().append_op(
                type="split",
                inputs={"X": orig_var},
                outputs={"Out": split_outs},
                attrs={"num": len(blocks)}  # assume split evenly
            )
        return splited

    def _concat_trainer_vars(self, program, splited):
        for varname, to_merge_list in splited.iteritems():
            orig_var = program.global_block().vars[varname]
            program.global_block().append_op(
                type="concat",
                inputs={"X": to_merge_list},
                outputs={"Out": orig_var},
                attrs={})

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
