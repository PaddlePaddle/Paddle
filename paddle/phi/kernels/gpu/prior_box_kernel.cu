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

#include <algorithm>
#include <vector>

#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T>
__device__ inline T clip(T in) {
  return min(max(in, 0.), 1.);
}

template <typename T>
__global__ void GenPriorBox(T* out,
                            const T* aspect_ratios,
                            const int height,
                            const int width,
                            const int im_height,
                            const int im_width,
                            const int as_num,
                            const T offset,
                            const T step_width,
                            const T step_height,
                            const T* min_sizes,
                            const T* max_sizes,
                            const int min_num,
                            bool is_clip,
                            bool min_max_aspect_ratios_order) {
  int num_priors = max_sizes ? as_num * min_num + min_num : as_num * min_num;
  int box_num = height * width * num_priors;
  CUDA_KERNEL_LOOP(i, box_num) {
    int h = i / (num_priors * width);
    int w = (i / num_priors) % width;
    int p = i % num_priors;
    int m = max_sizes ? p / (as_num + 1) : p / as_num;
    T cx = (w + offset) * step_width;
    T cy = (h + offset) * step_height;
    T bw, bh;
    T min_size = min_sizes[m];
    if (max_sizes) {
      int s = p % (as_num + 1);
      if (!min_max_aspect_ratios_order) {
        if (s < as_num) {
          T ar = aspect_ratios[s];
          bw = min_size * sqrt(ar) / 2.;
          bh = min_size / sqrt(ar) / 2.;
        } else {
          T max_size = max_sizes[m];
          bw = sqrt(min_size * max_size) / 2.;
          bh = bw;
        }
      } else {
        if (s == 0) {
          bw = bh = min_size / 2.;
        } else if (s == 1) {
          T max_size = max_sizes[m];
          bw = sqrt(min_size * max_size) / 2.;
          bh = bw;
        } else {
          T ar = aspect_ratios[s - 1];
          bw = min_size * sqrt(ar) / 2.;
          bh = min_size / sqrt(ar) / 2.;
        }
      }
    } else {
      int s = p % as_num;
      T ar = aspect_ratios[s];
      bw = min_size * sqrt(ar) / 2.;
      bh = min_size / sqrt(ar) / 2.;
    }
    T xmin = (cx - bw) / im_width;
    T ymin = (cy - bh) / im_height;
    T xmax = (cx + bw) / im_width;
    T ymax = (cy + bh) / im_height;
    out[i * 4] = is_clip ? clip<T>(xmin) : xmin;
    out[i * 4 + 1] = is_clip ? clip<T>(ymin) : ymin;
    out[i * 4 + 2] = is_clip ? clip<T>(xmax) : xmax;
    out[i * 4 + 3] = is_clip ? clip<T>(ymax) : ymax;
  }
}

template <typename T>
__global__ void SetVariance(T* out,
                            const T* var,
                            const int vnum,
                            const int num) {
  CUDA_KERNEL_LOOP(i, num) { out[i] = var[i % vnum]; }
}

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

  auto im_width = image.dims()[3];
  auto im_height = image.dims()[2];

  auto width = input.dims()[3];
  auto height = input.dims()[2];

  T step_width, step_height;
  if (new_step_w == 0 || new_step_h == 0) {
    step_width = static_cast<T>(im_width) / width;
    step_height = static_cast<T>(im_height) / height;
  } else {
    step_width = new_step_w;
    step_height = new_step_h;
  }

  int num_priors = new_aspect_ratios.size() * min_sizes.size();
  if (max_sizes.size() > 0) {
    num_priors += max_sizes.size();
  }
  int min_num = static_cast<int>(min_sizes.size());
  int box_num = width * height * num_priors;

  int block = 512;
  int grid = (box_num + block - 1) / block;

  auto stream = ctx.stream();

  ctx.template Alloc<T>(out);
  ctx.template Alloc<T>(var);

  DenseTensor r;
  paddle::framework::TensorFromVector(new_aspect_ratios, ctx, &r);

  DenseTensor min;
  paddle::framework::TensorFromVector(min_sizes, ctx, &min);

  T* max_data = nullptr;
  DenseTensor max;
  if (max_sizes.size() > 0) {
    paddle::framework::TensorFromVector(max_sizes, ctx, &max);
    max_data = max.data<T>();
  }

  GenPriorBox<T><<<grid, block, 0, stream>>>(out->data<T>(),
                                             r.data<T>(),
                                             height,
                                             width,
                                             im_height,
                                             im_width,
                                             new_aspect_ratios.size(),
                                             new_offset,
                                             step_width,
                                             step_height,
                                             min.data<T>(),
                                             max_data,
                                             min_num,
                                             clip,
                                             min_max_aspect_ratios_order);

  DenseTensor v;
  paddle::framework::TensorFromVector(variances, ctx, &v);
  grid = (box_num * 4 + block - 1) / block;
  SetVariance<T><<<grid, block, 0, stream>>>(
      var->data<T>(), v.data<T>(), variances.size(), box_num * 4);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    prior_box, GPU, ALL_LAYOUT, phi::PriorBoxKernel, float, double) {}
