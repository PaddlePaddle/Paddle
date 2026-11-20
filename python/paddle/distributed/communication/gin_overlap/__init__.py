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

"""gin_overlap 的通信原语：sharding stage1 的梯度桶本来每个一次 reduce_scatter，
这里改成用 NCCL 的 GIN 单边写（put）分几步做，每一步都能塞进反向里 DeepEP 不占网 /
不占 SM 的空档里。

这个包里只有原语：窗口 / 接收池 / host window 的布局、两个通信组的建立和校验、交给
C++ 侧放行层的每桶参数表，以及 2D 桶 gather 到 owner 的表（见 engine.py 的
GinOverlapPrimitive）。**什么时候**推进哪个阶段、桶队列、尾部交接、和 sharding 优化器
对象打交道的那部分都不在这里 —— 那是训练框架的事（PaddleFleet 的
paddlefleet/distributed/gin_overlap，它的 GinOverlapEngine 继承这里的
GinOverlapPrimitive）。

C++ / CUDA 实现在 paddle/fluid/distributed/collective/gin_overlap/，导出成
paddle.base.core.gin_overlap（paddle/fluid/pybind/gin_overlap_api.cc）；构建没开时
那个属性不存在，device_engine() 会带着原因报错（见 engine._BUILD_HINT）。

不是公开 API：__all__ 故意是空的，用的人按名字从 .engine 里取。
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
