# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from .fs import LocalFS  # noqa: F401
from .fs import HDFSClient  # noqa: F401
from .ps_util import DistributedInfer  # noqa: F401
import paddle.utils.deprecated as deprecated
from paddle.distributed import fleet

import paddle
from . import log_util  # noqa: F401
from . import hybrid_parallel_util  # noqa: F401

__all__ = [  #noqa
    "LocalFS", "recompute", "DistributedInfer", "HDFSClient"
]


@deprecated(since="2.4.0",
            update_to="paddle.distributed.fleet.recompute",
            level=1,
            reason="Please use new recompute API(fleet.recompute) ")
def recompute(function, *args, **kwargs):
    return fleet.recompute.recompute(function, *args, **kwargs)
