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

#include <vector>
#include "paddle/fluid/operators/math/concat_and_split.h"
#include "paddle/fluid/operators/stack_op.h"
#include "paddle/fluid/operators/strided_memcpy.h"

namespace paddle {
namespace operators {

template <typename T>
class StackKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto ins = ctx.MultiInput<framework::Tensor>("X");
    auto* out = ctx.Output<framework::Tensor>("Y");

    // Call concat functor
    int axis = ctx.Attr<int>("axis");
    axis = (axis > 0) ? axis : axis + ins[0]->dims().size();

    out->mutable_data<T>(ctx.GetPlace());

    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();

    // Sometimes direct copies will be faster, this maybe need deeply analysis.
    if (axis == 0 && ins.size() < 10) {
      size_t output_offset = 0;
      for (auto* in : ins) {
        if (!in || in->numel() == 0UL) {
          continue;
        }
        auto in_stride = framework::stride_numel(in->dims());
        auto out_stride = framework::stride_numel(out->dims());
        StridedNumelCopyWithAxis<T>(ctx.device_context(), axis,
                                    out->data<T>() + output_offset, out_stride,
                                    in->data<T>(), in_stride, in_stride[axis]);
        output_offset += in_stride[axis];
      }
    } else {
      std::vector<framework::Tensor> inputs;
      for (size_t i = 0; i < ins.size(); ++i) {
        if (ins[i] && ins[i]->numel() > 0) {
          inputs.push_back(*ins[i]);
        } else {
          continue;
        }
      }
      math::ConcatFunctor<platform::CUDADeviceContext, T> concat_functor;
      concat_functor(dev_ctx, inputs, static_cast<int>(axis), out);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace plat = paddle::platform;
namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(
    stack, ops::StackKernel<plat::CUDADeviceContext, float>,
    ops::StackKernel<plat::CUDADeviceContext, double>,
    ops::StackKernel<plat::CUDADeviceContext, int>,
    ops::StackKernel<plat::CUDADeviceContext, int64_t>,
    ops::StackKernel<plat::CUDADeviceContext, plat::float16>);

REGISTER_OP_CUDA_KERNEL(
    stack_grad, ops::StackGradKernel<plat::CUDADeviceContext, float>,
    ops::StackGradKernel<plat::CUDADeviceContext, double>,
    ops::StackGradKernel<plat::CUDADeviceContext, int>,
    ops::StackGradKernel<plat::CUDADeviceContext, int64_t>,
    ops::StackGradKernel<plat::CUDADeviceContext, plat::float16>);
