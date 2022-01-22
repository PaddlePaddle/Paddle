// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/seed_op.h"

namespace paddle {
namespace operators {

template <typename Place, typename T>
class GPUSeedKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *out = context.Output<Tensor>("Out");
    int seed = get_seed(context);

    auto force_cpu = context.Attr<bool>("force_cpu");
    bool cpu_place = force_cpu || context.GetPlace() == platform::CPUPlace();
    if (cpu_place) {
      platform::DeviceContextPool &pool =
          platform::DeviceContextPool::Instance();
      auto &dev_ctx = *pool.Get(platform::CPUPlace());
      out->mutable_data<T>(platform::CPUPlace());
      math::SetConstant<platform::CPUDeviceContext, T> functor;
      functor(reinterpret_cast<const platform::CPUDeviceContext &>(dev_ctx),
              out, static_cast<T>(seed));
    } else {
      auto *out_data = out->mutable_data<T>(context.GetPlace());
      auto target_gpu_place = context.GetPlace();
      auto stream = context.cuda_device_context().stream();
      memory::Copy(target_gpu_place, out_data, platform::CPUPlace(), &seed,
                   sizeof(int), stream);
    }
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(
    seed,
    paddle::operators::GPUSeedKernel<paddle::platform::CUDADeviceContext, int>);
