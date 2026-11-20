# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""gin_overlap 的通信原语：用 NCCL GIN 单边写分阶段完成 sharding stage1 每个梯度桶的 reduce_scatter。

这个包只提供原语：window / 接收池 / host window 的布局、两个通信组的建立和校验、交给 C++ 侧放行层的参数表，
以及 2D 桶 gather 到 owner 的表（engine.GinOverlapPrimitive，调用方继承它）。什么时候推进哪个阶段由调用方决定。
术语和布局的完整说明见 engine.py 的模块 docstring。

C++ / CUDA 实现在 paddle/fluid/distributed/collective/gin_overlap/，导出成 paddle.base.core.gin_overlap。
构建没开启时没有这个属性，device_engine() 会报错并给出构建条件。

不是公开 API：__all__ 是空的，按名字从这里或 .engine 取用。
"""

from .engine import (  # noqa: F401
    GROUP_DENSE,
    GROUP_MOE,
    LOG_EVERY,
    TIMEOUT_S,
    GinOverlapPrimitive,
    check_cuda,
    device_engine,
    log,
)

__all__ = []
