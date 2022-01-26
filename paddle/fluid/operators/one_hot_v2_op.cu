//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/one_hot_v2_op.h"
#include "paddle/pten/kernels/one_hot_kernel.h"

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;
template <typename DeviceContext, typename T>
class OneHotV2CUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in = context.Input<LoDTensor>("X");
    auto* out = context.Output<LoDTensor>("Out");
    int depth = context.Attr<int>("depth");
    bool allow_out_of_range = context.Attr<bool>("allow_out_of_range");
    auto depth_tensor = context.Input<LoDTensor>("depth_tensor");
    paddle::optional<const pten::DenseTensor&> depth_opt = paddle::none;
    if (depth_tensor != nullptr) {
      depth_opt = *depth_tensor;
    }

    auto& dev_ctx = context.device_context<DeviceContext>();
    pten::OneHotKernel<T>(
        static_cast<const typename framework::ConvertToPtenContext<
            DeviceContext>::TYPE&>(dev_ctx),
        *in, depth_opt, depth, context.Attr<int>("dtype"), allow_out_of_range,
        out);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    one_hot_v2,
    ops::OneHotV2CUDAKernel<paddle::platform::CUDADeviceContext, int>,
    ops::OneHotV2CUDAKernel<paddle::platform::CUDADeviceContext, int64_t>);
