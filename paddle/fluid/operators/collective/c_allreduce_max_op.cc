/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/collective/c_allreduce_op.h"

namespace paddle::framework {
class OpDesc;
template <typename T>
class EmptyGradOpMaker;
}  // namespace paddle::framework
namespace paddle::imperative {
class OpBase;
}  // namespace paddle::imperative

namespace paddle::operators {

class CAllReduceMaxOpMaker : public CAllReduceOpMaker {
 protected:
  std::string GetName() const override { return "Max"; }
};

DECLARE_INPLACE_OP_INFERER(AllreduceMaxInplaceInferer, {"X", "Out"});

DEFINE_C_ALLREDUCE_CPU_KERNEL(CAllReduceMax, kRedMax)

}  // namespace paddle::operators

namespace ops = paddle::operators;

REGISTER_OP_WITHOUT_GRADIENT(c_allreduce_max,
                             ops::CAllReduceOp,
                             ops::CAllReduceMaxOpMaker,
                             ops::AllreduceMaxInplaceInferer)
PD_REGISTER_STRUCT_KERNEL(c_allreduce_max,
                          CPU,
                          ALL_LAYOUT,
                          ops::CAllReduceMaxCPUKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          phi::dtype::float16) {}
