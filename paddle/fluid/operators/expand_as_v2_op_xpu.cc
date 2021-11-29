/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PADDLE_WITH_XPU

#include "paddle/fluid/operators/expand_as_v2_op.h"

namespace paddle {
namespace operators {

template <typename T>
class ExpandAsV2XPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto rank = context.Input<Tensor>("X")->dims().size();
    auto target_shape = context.Attr<std::vector<int>>("target_shape");
    auto target_rank = target_shape.size();
    PADDLE_ENFORCE_GE(target_rank, rank,
                      platform::errors::InvalidArgument(
                          "The rank (%d) of the input 'target_tensor' for "
                          "expand_as_v2 op must be greater than or equal to "
                          "the rank (%d) of the input 'x'.",
                          target_rank, rank));
    PADDLE_ENFORCE_GE(rank, 1, platform::errors::InvalidArgument(
                                   "The rank (%d) of the input 'x' for "
                                   "expand_as_v2 op must be positive.",
                                   rank));
    PADDLE_ENFORCE_LE(target_rank, MAX_RANK_SUPPORTED,
                      platform::errors::InvalidArgument(
                          "The rank (%d) of the input 'target_tensor' for "
                          "expand_as_v2 op must be less than or equal to %d.",
                          target_rank, MAX_RANK_SUPPORTED));
    ExpandAs(context);
  }

 protected:
  void ExpandAs(const framework::ExecutionContext& context) const {
    auto* in0 = context.Input<framework::Tensor>("X");
    auto in_dims = in0->dims();
    auto target_shape = context.Attr<std::vector<int>>("target_shape");
    auto vec_in_dims = framework::vectorize<int>(in_dims);
    auto diff = target_shape.size() - vec_in_dims.size();
    vec_in_dims.insert(vec_in_dims.begin(), diff, 1);

    for (size_t i = 0; i < vec_in_dims.size(); ++i) {
      PADDLE_ENFORCE_NE(target_shape[i], 0,
                        platform::errors::InvalidArgument(
                            "The value of target shape cannot be zero."));
      if (vec_in_dims[i] != 1) {
        PADDLE_ENFORCE_EQ(
            vec_in_dims[i], target_shape[i],
            platform::errors::InvalidArgument(
                "The value (%d) of the non-singleton dimension does not match"
                " the corresponding value (%d) in "
                "target tensor for expand_as_v2 op.",
                vec_in_dims[i], target_shape[i]));
      }
    }
    auto* out0 = context.Output<framework::Tensor>("Out");
    framework::DDim out_dims = framework::make_ddim(target_shape);
    out0->Resize(out_dims);
    out0->mutable_data<T>(context.GetPlace());
    auto& in0_shape = vec_in_dims;
    auto out0_shape = framework::vectorize<int>(out_dims);

    const auto& dev_ctx =
        context.template device_context<paddle::platform::XPUDeviceContext>();
    int r = XPU_SUCCESS;

    if (std::is_same<T, bool>::value) {
      auto in0_data = reinterpret_cast<const int8_t*>(in0->data<T>());
      auto out0_data = reinterpret_cast<int8_t*>(out0->data<T>());
      r = xpu::broadcast<int8_t>(dev_ctx.x_context(), in0_data, out0_data,
                                 in0_shape, out0_shape);
    } else {
      auto in0_data = reinterpret_cast<const XPUType*>(in0->data<T>());
      auto out0_data = reinterpret_cast<XPUType*>(out0->data<T>());
      r = xpu::broadcast<XPUType>(dev_ctx.x_context(), in0_data, out0_data,
                                  in0_shape, out0_shape);
    }
    PADDLE_ENFORCE_EQ(
        r, XPU_SUCCESS,
        platform::errors::External("XPU API(broadcast) return wrong "
                                   "value[%d %s] in ExpandAsV2XPUKernel.",
                                   r, XPUAPIErrorMsg[r]));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(expand_as_v2, ops::ExpandAsV2XPUKernel<float>,
                       ops::ExpandAsV2XPUKernel<paddle::platform::float16>,
                       ops::ExpandAsV2XPUKernel<bool>,
                       ops::ExpandAsV2XPUKernel<int>,
                       ops::ExpandAsV2XPUKernel<int64_t>);

#endif
