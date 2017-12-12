import framework
from backward import append_backward_ops
from regularizer import append_regularization_ops
import optimizer
from layer_helper import LayerHelper


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
