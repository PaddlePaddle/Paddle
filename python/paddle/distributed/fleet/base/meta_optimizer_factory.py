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

<<<<<<< HEAD
from ..meta_optimizers import *  # noqa: F401, F403
=======
from ..meta_optimizers import *
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

__all__ = []

meta_optimizer_names = list(
<<<<<<< HEAD
    filter(lambda name: name.endswith("Optimizer"), dir())
)
=======
    filter(lambda name: name.endswith("Optimizer"), dir()))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

# Because HybridParallelOptimizer is dygraph optimizer, it
# should be removed
meta_optimizer_names.remove("HybridParallelOptimizer")
meta_optimizer_names.remove("HeterParallelOptimizer")
<<<<<<< HEAD
meta_optimizer_names.remove("DGCMomentumOptimizer")


class MetaOptimizerFactory:
=======


class MetaOptimizerFactory(object):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def __init__(self):
        pass

    def _get_valid_meta_optimizers(self, user_defined_optimizer):
        opt_list = []
        for opt_name in meta_optimizer_names:
            opt_list.append(globals()[opt_name](user_defined_optimizer))
        return opt_list
