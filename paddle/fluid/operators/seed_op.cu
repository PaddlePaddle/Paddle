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

#include "paddle/fluid/operators/seed_op.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

template <typename T, typename DeviceContext>
class GPUSeedKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *out = context.Output<phi::DenseTensor>("Out");
    int seed = get_seed(context);

    auto force_cpu = context.Attr<bool>("force_cpu");
    bool cpu_place = force_cpu || context.GetPlace() == phi::CPUPlace();
    if (cpu_place) {
      phi::DeviceContextPool &pool = phi::DeviceContextPool::Instance();
      auto &dev_ctx = *pool.Get(phi::CPUPlace());
      out->mutable_data<T>(phi::CPUPlace());
      phi::funcs::SetConstant<phi::CPUContext, T> functor;
      functor(reinterpret_cast<const phi::CPUContext &>(dev_ctx),
              out,
              static_cast<T>(seed));
    } else {
      auto *out_data = out->mutable_data<T>(context.GetPlace());
      auto target_gpu_place = context.GetPlace();
      auto stream = context.cuda_device_context().stream();
      phi::memory_utils::Copy(target_gpu_place,
                              out_data,
                              phi::CPUPlace(),
                              &seed,
                              sizeof(int),
                              stream);
    }
  }
};

}  // namespace operators
}  // namespace paddle

PD_REGISTER_STRUCT_KERNEL(
    seed, GPU, ALL_LAYOUT, paddle::operators::GPUSeedKernel, int) {}
