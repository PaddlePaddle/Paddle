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

#include "paddle/phi/kernels/prior_box_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

template <typename T, typename Context>
void PriorBoxKernel(const Context& ctx,
                    const DenseTensor& input,
                    const DenseTensor& image,
                    const std::vector<float>& min_sizes,
                    const std::vector<float>& aspect_ratios,
                    const std::vector<float>& variances,
                    const std::vector<float>& max_sizes,
                    bool flip,
                    bool clip,
                    float step_w,
                    float step_h,
                    float offset,
                    bool min_max_aspect_ratios_order,
                    DenseTensor* out,
                    DenseTensor* var) {
  std::vector<float> new_aspect_ratios;
  ExpandAspectRatios(aspect_ratios, flip, &new_aspect_ratios);

  T new_step_w = static_cast<T>(step_w);
  T new_step_h = static_cast<T>(step_h);
  T new_offset = static_cast<T>(offset);

  auto img_width = image.dims()[3];
  auto img_height = image.dims()[2];

  auto feature_width = input.dims()[3];
  auto feature_height = input.dims()[2];

  T step_width, step_height;
  if (new_step_w == 0 || new_step_h == 0) {
    step_width = static_cast<T>(img_width) / feature_width;
    step_height = static_cast<T>(img_height) / feature_height;
  } else {
    step_width = new_step_w;
    step_height = new_step_h;
  }

  int num_priors = new_aspect_ratios.size() * min_sizes.size();
  if (max_sizes.size() > 0) {
    num_priors += max_sizes.size();
  }

  ctx.template Alloc<T>(out);
  ctx.template Alloc<T>(var);

  auto boxes_data = out->data<T>();
  auto var_data = var->data<T>();
  xpu::VectorParam<float> aspect_ratios_param{
      new_aspect_ratios.data(),
      static_cast<int>(new_aspect_ratios.size()),
      nullptr};
  xpu::VectorParam<float> min_sizes_param{
      min_sizes.data(), static_cast<int>(min_sizes.size()), nullptr};
  xpu::VectorParam<float> max_sizes_param{
      max_sizes.data(), static_cast<int>(max_sizes.size()), nullptr};

  int ret = xpu::gen_prior_box(ctx.x_context(),
                               boxes_data,
                               aspect_ratios_param,
                               min_sizes_param,
                               max_sizes_param,
                               feature_height,
                               feature_width,
                               img_height,
                               img_width,
                               new_offset,
                               step_height,
                               step_width,
                               clip,
                               min_max_aspect_ratios_order);
  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "gen_prior_box");

  int box_num = feature_height * feature_width * num_priors;
  int vlen = variances.size();
  std::vector<T> var_cpu(vlen * box_num);
  for (int i = 0; i < box_num; ++i) {
    std::copy(variances.begin(), variances.end(), var_cpu.begin() + i * vlen);
  }
  ctx.Wait();
  ret = xpu_memcpy(var_data,
                   var_cpu.data(),
                   var_cpu.size() * sizeof(T),
                   XPUMemcpyKind::XPU_HOST_TO_DEVICE);
  PADDLE_ENFORCE_XPU_SUCCESS(ret);
}

}  // namespace phi

PD_REGISTER_KERNEL(prior_box, XPU, ALL_LAYOUT, phi::PriorBoxKernel, float) {}
