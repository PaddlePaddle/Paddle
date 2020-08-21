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
namespace operators {

class CReduceMaxOpMaker : public CReduceOpMaker {
 protected:
  std::string GetName() const override { return "Max"; }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_WITHOUT_GRADIENT(c_reduce_max, ops::CReduceOp,
                             ops::CReduceMaxOpMaker);

REGISTER_OP_CPU_KERNEL(c_reduce_max,
                       ops::CReduceOpCPUKernel<ops::kRedMax, float>,
                       ops::CReduceOpCPUKernel<ops::kRedMax, double>,
                       ops::CReduceOpCPUKernel<ops::kRedMax, int>,
                       ops::CReduceOpCPUKernel<ops::kRedMax, int64_t>,
                       ops::CReduceOpCPUKernel<ops::kRedMax, plat::float16>);
