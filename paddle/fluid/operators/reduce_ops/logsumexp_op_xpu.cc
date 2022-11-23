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

#ifdef PADDLE_WITH_XPU

#include "paddle/fluid/operators/reduce_ops/reduce_op_function.h"
#include "paddle/fluid/platform/device/xpu/xpu_header.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class XPULogsumexpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input = context.Input<phi::DenseTensor>("X");
    auto* output = context.Output<phi::DenseTensor>("Out");

    auto axis = context.Attr<std::vector<int>>("axis");
    auto reduce_all = context.Attr<bool>("reduce_all");

    const auto& input_dim_size = input->dims().size();
    // The dims has full dim, set the reduce_all is True
    reduce_all |= (static_cast<const int>(axis.size()) == input_dim_size);

    const T* input_data = input->data<T>();
    T* output_data = output->mutable_data<T>(context.GetPlace());

    std::vector<int> axis_shape;
    std::vector<int> xdims(input_dim_size);
    for (int i = 0; i < input_dim_size; ++i) {
      xdims[i] = input->dims()[i];
    }
    if (reduce_all) {
      for (int i = 0; i < input_dim_size; ++i) {
        axis_shape.push_back(i);
      }
    } else {
      for (size_t i = 0; i < axis.size(); ++i) {
        int rdim = axis[i] < 0 ? axis[i] + input_dim_size : axis[i];
        axis_shape.push_back(rdim);
      }
    }

    auto& dev_ctx = context.template device_context<DeviceContext>();
    int r = xpu::logsumexp<T>(
        dev_ctx.x_context(), input_data, output_data, xdims, axis_shape);
    PADDLE_ENFORCE_EQ(r,
                      xpu::Error_t::SUCCESS,
                      platform::errors::External(
                          "XPU logsumexp kernel error! error value[%d %]",
                          r,
                          XPUAPIErrorMsg[r]));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(
    logsumexp,
    ops::XPULogsumexpKernel<paddle::platform::XPUDeviceContext, float>);
#endif
