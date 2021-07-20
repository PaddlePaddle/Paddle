/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the Licnse. */

#include "paddle/fluid/operators/randperm_op.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

void RangeFun(const platform::Place& place, const aclrtStream& stream, int n,
              Tensor* y) {
  y->mutable_data<int>({n}, place);
  const auto& runner = NpuOpRunner("Range", {0, n, 1}, {*y}, {});
  runner.Run(stream);
}
template <typename T>
class RandPermNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {}
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(randperm, ops::RandPermNPUKernel<float>,
                       ops::RandPermNPUKernel<double>,
                       ops::RandPermNPUKernel<plat::float16>,
                       ops::RandPermNPUKernel<int>,
                       ops::RandPermNPUKernel<int64_t>);
