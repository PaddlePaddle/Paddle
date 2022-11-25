// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device/device_wrapper.h"
#include "paddle/fluid/platform/device/xpu/xpu_header.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class GridSamplerXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_xpu_place(context.GetPlace()),
        true,
        platform::errors::Unavailable("This kernel only runs on XPU."));

    // input and output data
    const Tensor* input = context.Input<Tensor>("X");
    const Tensor* grid = context.Input<Tensor>("Grid");
    Tensor* output = context.Output<Tensor>("Output");

    int n = input->dims()[0];
    int c = input->dims()[1];
    int h = input->dims()[2];
    int w = input->dims()[3];
    int out_h = grid->dims()[1];
    int out_w = grid->dims()[2];

    // attrs
    // paddle.nn.functional.grid_sample(x, grid, mode='bilinear',
    // padding_mode='zeros', align_corners=True, name=None)
    const std::string mode = context.Attr<std::string>("mode");
    const std::string padding_mode = context.Attr<std::string>("padding_mode");
    bool align_corners_bool = context.Attr<bool>("align_corners");
    const std::string data_format =
        paddle::framework::DataLayoutToString(input->layout());

    // attr to real param
    bool is_nearest_bool;
    if (mode == "bilinear") {
      is_nearest_bool = false;
    } else if (mode == "nearest") {
      is_nearest_bool = true;
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "should not reach here: mode should be either 'bilinear' or "
          "'nearest', bot got %s.",
          mode));
    }

    // attention: 0: zeros, 2: reflection, 1: border according to XDNN api.
    int padding_mode_int;
    if (padding_mode == "zeros") {
      padding_mode_int = 0;
    } else if (padding_mode == "reflection") {
      padding_mode_int = 2;
    } else if (padding_mode == "border") {
      padding_mode_int = 1;
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "should not reach here: padding_mode should be either 'zeros' or "
          "'reflection' or 'border', bot got %s.",
          padding_mode));
    }

    bool is_nchw_bool;
    if (data_format == "NCHW") {
      is_nchw_bool = true;
    } else if (data_format == "NHWC") {
      is_nchw_bool = false;
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "should not reach here: data_format should be either 'NCHW' or "
          "'NHWC', bot got %s.",
          data_format));
    }

    // data pointers
    const T* input_data = input->data<T>();
    const T* grid_data = grid->data<T>();
    T* output_data =
        output->mutable_data<T>({n, c, out_h, out_w}, context.GetPlace());

    auto& dev_ctx = context.template device_context<DeviceContext>();
    // int grid_sample(Context* ctx, const T* x, const T* grid, T* y, int n, int
    // c, int xh, int xw, int yh, int yw, bool is_nearest, bool align_corners,
    // int padding_mode, bool is_nchw);
    int r = xpu::grid_sample(dev_ctx.x_context(),
                             input_data,
                             grid_data,
                             output_data,
                             n,
                             c,
                             h,
                             w,
                             out_h,
                             out_w,
                             is_nearest_bool,
                             align_corners_bool,
                             padding_mode_int,
                             is_nchw_bool);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "grid_sampler");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_XPU_KERNEL(
    grid_sampler,
    ops::GridSamplerXPUKernel<paddle::platform::XPUDeviceContext, float>);

#endif
