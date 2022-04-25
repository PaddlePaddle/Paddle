// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/data/batch_resize_op.h"
#include "paddle/fluid/operators/interpolate_op.cu.h"
#include "paddle/fluid/platform/device/gpu/gpu_launch_config.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"

namespace paddle {
namespace operators {
namespace data {

using DataLayout = framework::DataLayout;

template <typename T>
static void ResizeFwd(const framework::ExecutionContext& ctx,
                      const framework::Tensor& input, framework::Tensor* output,
                      const std::vector<int64_t> out_size,
                      const std::string interp_method, const bool align_corners,
                      const int align_mode, const int img_h, const int img_w,
                      const int c, const DataLayout data_format) {
  auto input_data = input.template data<T>();
  int out_h = static_cast<int>(out_size[0]);
  int out_w = static_cast<int>(out_size[1]);

  framework::DDim dim_out;
  if (data_format == DataLayout::kNCHW) {
    dim_out = {c, out_h, out_w};
  } else {
    dim_out = {out_h, out_w, c};
  }
  auto output_data = output->data<T>();

  float ratio_h = 0.f;
  float ratio_w = 0.f;
  if (out_h > 1) {
    ratio_h = (align_corners) ? static_cast<float>(img_h - 1) / (out_h - 1)
                              : static_cast<float>(img_h) / out_h;
  }
  if (out_w > 1) {
    ratio_w = (align_corners) ? static_cast<float>(img_w - 1) / (out_w - 1)
                              : static_cast<float>(img_w) / out_w;
  }

  int in_chw = c * img_h * img_w;
  int out_chw = c * out_h * out_w;

  platform::GpuLaunchConfig config =
      platform::GetGpuLaunchConfig1D(ctx.cuda_device_context(), out_chw);

  if ("nearest" == interp_method) {
    KeNearestNeighborInterpFw<
        T><<<config.block_per_grid, config.thread_per_block, 0,
             ctx.cuda_device_context().stream()>>>(
        input_data, img_h, img_w, 1, in_chw, output_data, out_h, out_w, 1,
        out_chw, c, ratio_h, ratio_w, align_corners, data_format);
  } else if ("bilinear" == interp_method) {
    KeBilinearInterpFw<T><<<config.block_per_grid, config.thread_per_block, 0,
                            ctx.cuda_device_context().stream()>>>(
        input_data, img_h, img_w, 1, in_chw, output_data, out_h, out_w, 1,
        out_chw, c, ratio_h, ratio_w, align_corners, align_mode, data_format);
  }
}

template <typename T>
class BatchResizeCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(ctx.GetPlace()), true,
        platform::errors::NotFound("This kernel only runs on GPU device."));
    // get input, output
    auto x = ctx.MultiInput<framework::Tensor>("X");
    PADDLE_ENFORCE_GT(x.size(), 0,
                      platform::errors::InvalidArgument(
                          "The size of X must be greater than 0."));
    auto* out = ctx.Output<framework::Tensor>("Out");

    // get size, scale, ratio
    auto size = ctx.Attr<std::vector<int64_t>>("size");

    const std::string data_format_str = ctx.Attr<std::string>("data_format");
    const DataLayout data_format =
        framework::StringToDataLayout(data_format_str);
    // get interpolation method
    const std::string interp_method = ctx.Attr<std::string>("interp_method");
    bool align_corners = ctx.Attr<bool>("align_corners");
    int align_mode = ctx.Attr<int>("align_mode");

    auto* img = x.at(0);
    int64_t img_c =
        data_format == DataLayout::kNCHW ? img->dims()[0] : img->dims()[2];

    out->mutable_data<T>(ctx.GetPlace());

    int img_h, img_w, idx_h, idx_w, crop_h, crop_w;
    for (int i = 0; i < x.size(); i++) {
      img = x.at(i);
      img_h =
          data_format == DataLayout::kNCHW ? img->dims()[1] : img->dims()[0];
      img_w =
          data_format == DataLayout::kNCHW ? img->dims()[2] : img->dims()[1];
      auto out_tensor = out->Slice(i, i + 1);
      ResizeFwd<T>(ctx, *img, &out_tensor, size, interp_method, align_corners,
                   align_mode, img_h, img_w, img_c, data_format);
    }
  }
};

}  // namespace data
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(batch_resize, ops::data::BatchResizeCUDAKernel<uint8_t>,
                        ops::data::BatchResizeCUDAKernel<float>,
                        ops::data::BatchResizeCUDAKernel<double>);
