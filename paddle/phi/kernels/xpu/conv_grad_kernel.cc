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

#include "paddle/phi/kernels/conv_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/conv_util.h"

namespace phi {

template <typename T, typename Context>
void ConvGradKernel(const Context& dev_ctx,
                    const DenseTensor& input,
                    const DenseTensor& filter,
                    const DenseTensor& out_grad,
                    const std::vector<int>& strides,
                    const std::vector<int>& paddings_t,
                    const std::string& padding_algorithm,
                    int groups,
                    const std::vector<int>& dilations_t,
                    const std::string& data_format,
                    bool use_addto,
                    int workspace_size_MB,
                    bool exhaustive_search,
                    DenseTensor* input_grad,
                    DenseTensor* filter_grad) {
  using XPUT = typename XPUTypeTrait<T>::Type;
  std::vector<int> paddings = paddings_t;
  std::vector<int> dilations = dilations_t;
  // The filter and filter_grad will be reshaped in the calculations,
  // so here use an assignment operation,
  // that avoids modifying the variable in the Scope.
  if (!input_grad && !filter_grad) return;
  PADDLE_ENFORCE_EQ(
      data_format == "NDHWC",
      false,
      phi::errors::InvalidArgument(
          ("XPU doesn't support data_format is NDHWC in conv grad op.")));

  phi::DDim in_data_dims =
      phi::slice_ddim(input.dims(), 2, input.dims().size());
  phi::DDim filter_data_dims =
      phi::slice_ddim(filter.dims(), 2, filter.dims().size());
  std::vector<int> ksize = phi::vectorize<int>(filter_data_dims);
  std::vector<int> filter_shape = phi::vectorize<int>(filter.dims());
  UpdatePaddingAndDilation(
      &paddings, &dilations, padding_algorithm, in_data_dims, strides, ksize);

  int batch_size = static_cast<int>(input.dims()[0]);
  int img_c = static_cast<int>(input.dims()[1]);
  int img_h = static_cast<int>(input.dims()[2]);
  int img_w = static_cast<int>(input.dims()[3]);
  int f = static_cast<int>(filter.dims()[0]);
  bool is_nchw = true;
  if (data_format == "NHWC") {
    img_c = static_cast<int>(input.dims()[3]);
    img_h = static_cast<int>(input.dims()[1]);
    img_w = static_cast<int>(input.dims()[2]);
    is_nchw = false;
  }

  const XPUT* input_data = reinterpret_cast<const XPUT*>(input.data<T>());
  const XPUT* filter_data = reinterpret_cast<const XPUT*>(filter.data<T>());
  const XPUT* output_grad_data =
      reinterpret_cast<const XPUT*>(out_grad.data<T>());
  XPUT* input_grad_data = nullptr;
  if (input_grad) {
    dev_ctx.template Alloc<T>(input_grad);
    input_grad_data = reinterpret_cast<XPUT*>(input_grad->data<T>());
  }
  XPUT* filter_grad_data = nullptr;
  if (filter_grad) {
    dev_ctx.template Alloc<T>(filter_grad);
    filter_grad_data = reinterpret_cast<XPUT*>(filter_grad->data<T>());
  }
  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());

  XPUT* filter_data_tmp;
  XPUT* filter_grad_data_tmp;
  const XPUT* filter_data_ptr = filter_data;
  XPUT* filter_grad_data_ptr = filter_grad_data;
  if (data_format == "NHWC") {
    filter_data_tmp = RAII_GUARD.alloc<XPUT>(filter.numel());
    PADDLE_ENFORCE_XDNN_NOT_NULL(filter_data_tmp);
    int r = xpu::transpose<XPUT>(dev_ctx.x_context(),
                                 filter_data,
                                 filter_data_tmp,
                                 filter_shape,
                                 {0, 2, 3, 1});
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");
    filter_data_ptr = reinterpret_cast<const XPUT*>(filter_data_tmp);

    if (filter_grad_data != nullptr) {
      filter_grad_data_tmp = RAII_GUARD.alloc<XPUT>(filter.numel());
      PADDLE_ENFORCE_XDNN_NOT_NULL(filter_grad_data_tmp);
      filter_grad_data_ptr = filter_grad_data_tmp;
    }
  }
  int r = xpu::conv2d_grad<XPUT, XPUT, XPUT, int16_t>(dev_ctx.x_context(),
                                                      input_data,
                                                      filter_data_ptr,
                                                      output_grad_data,
                                                      input_grad_data,
                                                      filter_grad_data_ptr,
                                                      batch_size,
                                                      img_c,
                                                      img_h,
                                                      img_w,
                                                      f,
                                                      ksize,
                                                      strides,
                                                      paddings,
                                                      dilations,
                                                      groups,
                                                      nullptr,
                                                      nullptr,
                                                      nullptr,
                                                      nullptr,
                                                      nullptr,
                                                      is_nchw);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "conv2d_grad");

  if ((filter_grad_data_ptr != nullptr) && (data_format == "NHWC")) {
    std::vector<int> filter_shape_fhwc = {
        filter_shape[0], filter_shape[2], filter_shape[3], filter_shape[1]};
    int r = xpu::transpose<XPUT>(dev_ctx.x_context(),
                                 filter_grad_data_ptr,
                                 filter_grad_data,
                                 filter_shape_fhwc,
                                 {0, 3, 1, 2});
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");
  }
}

template <typename T, typename Context>
void DepthwiseConvGradKernel(const Context& dev_ctx,
                             const DenseTensor& input,
                             const DenseTensor& filter,
                             const DenseTensor& out_grad,
                             const std::vector<int>& strides,
                             const std::vector<int>& paddings,
                             const std::string& paddding_algorithm,
                             int groups,
                             const std::vector<int>& dilations,
                             const std::string& data_format,
                             bool use_addto,
                             int workspace_size_MB,
                             bool exhaustive_search,
                             bool fuse_relu,
                             DenseTensor* input_grad,
                             DenseTensor* filter_grad) {
  ConvGradKernel<T, Context>(dev_ctx,
                             input,
                             filter,
                             out_grad,
                             strides,
                             paddings,
                             paddding_algorithm,
                             groups,
                             dilations,
                             data_format,
                             use_addto,
                             workspace_size_MB,
                             exhaustive_search,
                             input_grad,
                             filter_grad);
}

}  // namespace phi

PD_REGISTER_KERNEL(conv2d_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::ConvGradKernel,
                   float,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(depthwise_conv2d_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::DepthwiseConvGradKernel,
                   float) {}
