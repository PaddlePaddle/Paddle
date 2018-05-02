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

from paddle.fluid.framework import Program, Variable, program_guard
from paddle.fluid.optimizer import Optimizer


def create_programs_from_network(network_func, optimizer=None):
    prog = Program()
    startup_prog = Program()

    fetch_params = []
    optimize_ops = None
    params_grads = None

    if network_func:
        with program_guard(prog, startup_prog):
            network_fetch_params = network_func()

            if isinstance(network_fetch_params, Variable):
                fetch_params = [network_fetch_params]
            elif isinstance(network_fetch_params, tuple):
                fetch_params = network_fetch_params
            else:
                raise TypeError("network function can only return a Variable "
                                "or tuple of Variables")

            if optimizer:
                if isinstance(optimizer, Optimizer):
                    if len(fetch_params) == 0:
                        raise Exception("network function must return a loss "
                                        "Variable")
                    loss_var = fetch_params[0]
                    optimize_ops, params_grads = optimizer.minimize(
                        loss_var, startup_prog)
                else:
                    raise TypeError("optimizer is not instance of Optimizer")

    return startup_prog, prog, fetch_params, optimize_ops, params_grads
