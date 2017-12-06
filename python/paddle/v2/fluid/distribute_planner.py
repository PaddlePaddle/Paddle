import framework
from backward import append_backward_ops
from regularizer import append_regularization_ops
import optimizer
from layer_helper import LayerHelper

__all__ = ['SGD', 'Momentum', 'Adagrad', 'Adam', 'Adamax', 'DecayedAdagrad']


def hash_name_to_server(parameters, pserver_endpoints):
    def _hash_param(param_name, total):
        return hash(param_name) % total

    param_map = dict()
    for param in parameters:
        if param.trainable is True:
            server_id = _hash_param(param.name, len(pserver_endpoints))
            server_for_param = pserver_endpoints[server_id]
            if param_map.has_key(server_for_param):
                param_map[server_for_param].append(param)
            else:
                param_map[server_for_param] = [param]

    return param_map


def round_robin(parameters, pserver_endpoints):
    assert (len(parameters) < len(pserver_endpoints))

    param_map = dict()
    pserver_idx = 0
    for param in parameters:
        if param.trainable is True:
            server_for_param = pserver_endpoints[pserver_idx]
            if param_map.has_key(server_for_param):
                param_map[server_for_param].append(param)
            else:
                param_map[server_for_param] = [param]

            pserver_idx += 1
            if pserver_idx > len(pserver_endpoints):
                pserver_idx = 0
    return param_map


def _append_sendop_for_trainer(loss,
                               parameters_and_grads,
                               pserver_endpoints,
                               split_method=round_robin):
    assert (callable(split_method))
    param_map, grad_map = \
        split_method(parameters_and_grads, pserver_endpoints)

    for ep in pserver_endpoints:
        # FIXME(typhoonzero): send to different servers can run in parrallel.
        send_op = loss.block.append_op(
            type="send",
            inputs={"X": param_map[ep]},
            outputs={"Out": param_map[ep]},
            attrs={"endpoint": ep})

    return send_op


class DistributedPlanner(optimizer.Optimizer):
    def __init__(self, global_step=None, parallelism_type='dp'):
        """
            parallelism_type:
                dp: data parallelism
                mp: model parallelism
        """
        super(DistributedPlanner).__init__(self, global_step)
        if parallelism_type == "mp":
            raise NotImplementedError("model parallelism not implemented")
        elif parallelism_type == "dp":
            self.parameter_server_program_map = dict()
            self.worker_program = None
        else:
            raise NameError("parallelism_type %s not supported" %
                            parallelism_type)

    def create_optimization_pass(self,
                                 parameters_and_grads,
                                 program,
                                 startup_program=None):
        # Create any accumulators
        self.helper = LayerHelper(
            self.__class__.__name__,
            main_program=program,
            startup_program=startup_program)
        self._create_accumulators(program.global_block(),
                                  [p[0] for p in parameters_and_grads])

        optimize_ops = []
        for param_and_grad in parameters_and_grads:
            if param_and_grad[0].trainable is True and param_and_grad[
                    1] is not None:
                optimize_op = self._append_optimize_op(program.global_block(),
                                                       param_and_grad)
                optimize_ops.append(optimize_op)

        # Returned list of ops can include more ops in addition
        # to optimization ops
        return_ops = optimize_ops

        # Get custom finish ops for subclasses
        # FIXME: Need to fix this once we figure out how to handle dependencies
        finish_ops = self._finish_update(program.global_block())
        if finish_ops is not None:
            return_ops += finish_ops

        if self._global_step is not None:
            return_ops.append(
                self._increment_global_step(program.global_block()))
        return return_ops

    def minimize(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None,
                 split_method=round_robin):
        """
            For distributed case, this call append backward ops and then
            append sevaral send_ops at the end for each parameter server.

            Then call get_pserver_program(idx/endpoint) will return the program of
            coresponding pserver program to run.
        """
        params_grads = append_backward_ops(loss, parameter_list, no_grad_set)
        # Add regularization if any
        params_grads = append_regularization_ops(params_grads)
        _append_sendop_for_trainer(loss, params_grads, self.pserver_endpoints,
                                   split_method)
        self.worker_program = loss.block.program

        optimize_sub_program = framework.Program()
        optimize_ops = self.create_optimization_pass(
            params_grads, optimize_sub_program, startup_program)
        param_list = []
        for param_and_grad in params_grads:
            if param_and_grad[0].trainable is True and param_and_grad[
                    1] is not None:
                param_list.append(param_and_grad[0])

        param_map, grad_map = \
            split_method(params_grads, self.pserver_endpoints)

        for ep in self.pserver_endpoints:
            pserver_program = framework.Program()
            self.parameter_server_program_map[ep] = pserver_program
            pserver_program.global_block().append_op(
                type="recv",
                inputs={"RX": param_map[ep]},
                outputs={},
                attrs={
                    "OptimizeBlock": optimize_sub_program.global_block(),
                    "endpoint": ep
                })
        # FIXME(typhoonzero): when to use this return value?
        return None

    def get_pserver_program(self, endpoint):
        return self.parameter_server_program_map.get(endpoint)


SGD = optimizer.SGDOptimizer
Momentum = optimizer.MomentumOptimizer
Adagrad = optimizer.AdagradOptimizer
Adam = optimizer.AdamOptimizer
Adamax = optimizer.AdamaxOptimizer
DecayedAdagrad = optimizer.DecayedAdagradOptimizer

for optcls in __all__:
    eval(optcls).__base__ = DistributedPlanner
