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

#include "paddle/fluid/operators/reduce_ops/reduce_prod_op.h"

namespace paddle {
namespace framework {
class OpDesc;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
namespace platform {
class CPUDeviceContext;
}  // namespace platform
}  // namespace paddle

REGISTER_REDUCE_OP(reduce_prod);
REGISTER_OP_CPU_KERNEL(reduce_prod,
                       ops::ReduceKernel<paddle::platform::CPUDeviceContext,
                                         float, ops::ProdFunctor>,
                       ops::ReduceKernel<paddle::platform::CPUDeviceContext,
                                         double, ops::ProdFunctor>,
                       ops::ReduceKernel<paddle::platform::CPUDeviceContext,
                                         int, ops::ProdFunctor>,
                       ops::ReduceKernel<paddle::platform::CPUDeviceContext,
                                         int64_t, ops::ProdFunctor>);
REGISTER_OP_CPU_KERNEL(reduce_prod_grad,
                       ops::ReduceGradKernel<paddle::platform::CPUDeviceContext,
                                             float, ops::ProdGradFunctor>,
                       ops::ReduceGradKernel<paddle::platform::CPUDeviceContext,
                                             double, ops::ProdGradFunctor>,
                       ops::ReduceGradKernel<paddle::platform::CPUDeviceContext,
                                             int, ops::ProdGradFunctor>,
                       ops::ReduceGradKernel<paddle::platform::CPUDeviceContext,
                                             int64_t, ops::ProdGradFunctor>);
