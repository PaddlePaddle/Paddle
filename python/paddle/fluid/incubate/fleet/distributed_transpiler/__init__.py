#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import sys

from paddle.fluid.executor import Executor

from ..base.fleet_base import Fleet
from ..base.fleet_base import DistributedOptimizer


class DistributedTranspiler(Fleet):
    def __init__(self, mode):
        super(DistributedTranspiler, self).__init__(mode)

    def init_worker(self, executor, startup_program=None):

        pass

    def run_worker(self, executor, main_program=None):
        pass

    def init_server(self, executor, startup_program=None, model_dir=None):
        pass

    def run_server(self, executor, main_program=None):
        pass

    def barrier_worker(self):
        pass

    def stop_worker(self):
        pass

    def stop(self):
        pass

    def distributed_optimizer(self, optimizer, strategy=None):
        if not isinstance(optimizer, Optimizer):
            raise ValueError("optimizer must be an instance of Optimizer")
        self.optimizer = TranspilerOptimizer(optimizer, strategy)
        return self.optimizer

    def save_inference_model(self,
                             executor,
                             dirname,
                             feeded_var_names,
                             target_vars,
                             main_program=None,
                             model_filename=None,
                             params_filename=None,
                             export_for_deployment=True):
        pass

    def save_persistables(self,
                          executor,
                          dirname,
                          main_program=None,
                          filename=None):
        pass


class TranspilerOptimizer(DistributedOptimizer):
    def __init__(self, optimizer, strategy=None):
        super(TranspilerOptimizer, self).__init__(optimizer, strategy)

    def backward(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None,
                 callbacks=None):
        pass

    def apply_gradients(self, params_grads):
        pass

    def minimize(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None):
        pass
