/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/clip_by_norm_op.h"

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(clip_by_norm, ops::ClipByNormOp,
                             ops::ClipByNormOpMaker);

REGISTER_OP_CPU_KERNEL(
    clip_by_norm,
    ops::ClipByNormKernel<paddle::platform::CPUDeviceContext, float>);
