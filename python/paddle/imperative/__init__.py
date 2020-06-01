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

# define api used to run in imperative mode 
__all__ = [
    'BackwardStrategy', 'enabled', 'grad', 'guard', 'LayerList', 'load', 'save',
    'prepare_context', 'to_variable', 'TracedLayer', 'no_grad', 'ParallelEnv',
    'ProgramTranslator', 'declarative', 'DataParallel'
]

__all__ += [
    'NoamDecay', 'PiecewiseDecay', 'NaturalExpDecay', 'ExponentialDecay',
    'InverseTimeDecay', 'PolynomialDecay', 'CosineDecay'
]

from paddle.fluid import core
from ..fluid.dygraph.base import enabled, guard, no_grad, to_variable, grad
from ..fluid.dygraph.checkpoint import load_dygraph as load
from ..fluid.dygraph.checkpoint import save_dygraph as save
from ..fluid.dygraph.parallel import prepare_context, ParallelEnv, DataParallel
from ..fluid.dygraph.jit import TracedLayer, declarative
from ..fluid.dygraph import ProgramTranslator

from ..fluid.dygraph.learning_rate_scheduler import NoamDecay, PiecewiseDecay, NaturalExpDecay, ExponentialDecay, \
        InverseTimeDecay, PolynomialDecay, CosineDecay

BackwardStrategy = core.BackwardStrategy
