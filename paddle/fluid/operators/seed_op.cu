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

#include <cuda.h>
#include "paddle/fluid/operators/seed_op.h"

namespace paddle {
namespace operators {

template <typename Place, typename T>
class GPUSeedKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* out = context.Output<Tensor>("Out");
    auto* out_data = out->mutable_data<T>(context.GetPlace());
    int user_seed = context.Attr<int>("seed");
    std::random_device rnd;
    int seed;
    if (user_seed != 0) {
      seed = user_seed;
    } else {
      seed = rnd();
    }
    auto target_gpu_place =
        BOOST_GET_CONST(platform::CUDAPlace, context.GetPlace());
    auto stream = context.cuda_device_context().stream();
    memory::Copy(target_gpu_place, out_data, platform::CPUPlace(), &seed,
                 sizeof(int), stream);
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(
    seed,
    paddle::operators::GPUSeedKernel<paddle::platform::CUDADeviceContext, int>);
