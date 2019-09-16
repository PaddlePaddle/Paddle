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
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/gpu_info.h"

namespace paddle {
namespace operators {
using platform::PADDLE_CUDA_NUM_THREADS;

template <typename InT, typename OutT>
__global__ void FillOutputKernel(const InT* p_in_data, OutT* p_out_data,
                                 const int64_t numel, const int depth) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel && p_in_data[idx] >= 0 && p_in_data[idx] < depth) {
    *(p_out_data + (idx * depth) + p_in_data[idx]) = 1.0;
  }
}

template <typename DeviceContext, typename InT>
struct OneHotV2OpCUDAFunctor {
  const framework::LoDTensor* in_;
  framework::LoDTensor* out_;
  const DeviceContext& ctx_;
  int depth_;

  OneHotV2OpCUDAFunctor(const framework::LoDTensor* in,
                        framework::LoDTensor* out, int depth,
                        const DeviceContext& ctx)
      : in_(in), out_(out), depth_(depth), ctx_(ctx) {}

  template <typename OutT>
  void apply() const {
    auto* p_in_data = in_->data<InT>();
    auto numel = in_->numel();
    auto* p_out_data = out_->mutable_data<OutT>(ctx_.GetPlace());
    auto stream = ctx_.stream();
    math::set_constant(ctx_, out_, 0.0);

    FillOutputKernel<<<(numel + PADDLE_CUDA_NUM_THREADS - 1) /
                           PADDLE_CUDA_NUM_THREADS,
                       PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
        p_in_data, p_out_data, numel, depth_);
  }
};

using LoDTensor = framework::LoDTensor;
template <typename DeviceContext, typename T>
class OneHotV2CUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in = context.Input<LoDTensor>("X");
    auto* out = context.Output<LoDTensor>("Out");

    int depth = -1;
    if (context.HasInput("depth_tensor")) {
      auto* depth_tensor = context.Input<framework::Tensor>("depth_tensor");
      if (platform::is_gpu_place(depth_tensor->place())) {
        framework::Tensor temp;
        TensorCopySync(*depth_tensor, platform::CPUPlace(), &temp);
        depth = *temp.data<int32_t>();
      } else {
        depth = *depth_tensor->data<int32_t>();
      }

      auto out_dims = out->dims();
      out_dims[out_dims.size() - 1] = depth;
      out->Resize(out_dims);
    } else {
      depth = context.Attr<int>("depth");
    }
    framework::VisitDataType(
        static_cast<framework::proto::VarType::Type>(
            context.Attr<int>("dtype")),
        OneHotV2OpCUDAFunctor<DeviceContext, T>(
            in, out, depth, context.template device_context<DeviceContext>()));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    one_hot_v2,
    ops::OneHotV2CUDAKernel<paddle::platform::CUDADeviceContext, int>,
    ops::OneHotV2CUDAKernel<paddle::platform::CUDADeviceContext, int64_t>);
