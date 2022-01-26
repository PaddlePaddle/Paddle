// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/reduce_ops/reduce_all_op.h"

namespace paddle {
namespace framework {
class OpDesc;
template <typename T>
class EmptyGradOpMaker;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
namespace platform {
class CPUDeviceContext;
}  // namespace platform
}  // namespace paddle

// kernel's device type is decided by input tensor place, to be consistent with
// compare and logical ops
REGISTER_REDUCE_OP_WITHOUT_GRAD(reduce_all, UseInputPlace);
REGISTER_OP_CPU_KERNEL(reduce_all,
                       ops::BoolReduceKernel<paddle::platform::CPUDeviceContext,
                                             bool, ops::AllFunctor>);
