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
#include "paddle/fluid/operators/split_op.h"
#include <string>
#include <vector>
#include "paddle/fluid/platform/device/xpu/xpu_header.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class SplitXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<framework::Tensor>("X");
    auto output = ctx.MultiOutput<framework::Tensor>("Out");
    int num = ctx.Attr<int>("num");
    std::vector<int> sections = ctx.Attr<std::vector<int>>("sections");
    int axis = ctx.Attr<int>("axis");
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    auto in_dims = input->dims();

    auto input_shape = framework::vectorize<int>(in_dims);
    std::vector<int> split_lists;
    std::vector<T*> out_ptrs;
    auto outs_number = output.size();
    std::vector<framework::DDim> outs_dims =
        UpdateOutsDims(true, true, in_dims, num, sections, axis, outs_number);
    for (size_t i = 0; i < output.size(); ++i) {
      output[i]->Resize(outs_dims[i]);
      out_ptrs.push_back(output[i]->mutable_data<T>(ctx.GetPlace()));
      split_lists.push_back(output[i]->dims()[axis]);
    }

    int r = xpu::split<T>(dev_ctx.x_context(), input->data<T>(), out_ptrs,
                          input_shape, split_lists, axis);
    PADDLE_ENFORCE_EQ(
        r, XPU_SUCCESS,
        platform::errors::External("XPU split kernel return wrong value[%d %s]",
                                   r, XPUAPIErrorMsg[r]));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(
    split, ops::SplitXPUKernel<paddle::platform::XPUDeviceContext, float>,
    ops::SplitXPUKernel<paddle::platform::XPUDeviceContext, int>);
#endif
