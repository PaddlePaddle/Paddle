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

#include "paddle/fluid/operators/collective/c_reduce_op.h"

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
struct CPUPlace;
struct float16;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace operators {

class CReduceSumOpMaker : public CReduceOpMaker {
 protected:
  std::string GetName() const override { return "Sum"; }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_WITHOUT_GRADIENT(c_reduce_sum, ops::CReduceOp,
                             ops::CReduceSumOpMaker);

REGISTER_OP_CPU_KERNEL(c_reduce_sum,
                       ops::CReduceOpCPUKernel<ops::kRedSum, float>,
                       ops::CReduceOpCPUKernel<ops::kRedSum, double>,
                       ops::CReduceOpCPUKernel<ops::kRedSum, int>,
                       ops::CReduceOpCPUKernel<ops::kRedSum, int64_t>,
                       ops::CReduceOpCPUKernel<ops::kRedSum, plat::float16>)
