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

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

template <typename T, typename Context>
void PriorBoxKernel(const Context& ctx,
                    const DenseTensor& input,
                    const DenseTensor& image,
                    const std::vector<float>& min_sizes,
                    const std::vector<float>& max_sizes,
                    const std::vector<float>& aspect_ratios,
                    const std::vector<float>& variances,
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

  int num_priors =
      static_cast<int>(new_aspect_ratios.size() * min_sizes.size());
  if (!max_sizes.empty()) {
    num_priors += static_cast<int>(max_sizes.size());
  }

  ctx.template Alloc<T>(out);
  ctx.template Alloc<T>(var);

  T* b_t = out->data<T>();
  for (int h = 0; h < feature_height; ++h) {
    for (int w = 0; w < feature_width; ++w) {
      T center_x = (w + new_offset) * step_width;
      T center_y = (h + new_offset) * step_height;
      T box_width, box_height;
      for (size_t s = 0; s < min_sizes.size(); ++s) {
        auto min_size = min_sizes[s];
        if (min_max_aspect_ratios_order) {
          box_width = box_height = min_size / 2.;
          b_t[0] = (center_x - box_width) / img_width;
          b_t[1] = (center_y - box_height) / img_height;
          b_t[2] = (center_x + box_width) / img_width;
          b_t[3] = (center_y + box_height) / img_height;
          b_t += 4;
          if (!max_sizes.empty()) {
            auto max_size = max_sizes[s];
            // square prior with size sqrt(minSize * maxSize)
            box_width = box_height = sqrt(min_size * max_size) / 2.;
            b_t[0] = (center_x - box_width) / img_width;
            b_t[1] = (center_y - box_height) / img_height;
            b_t[2] = (center_x + box_width) / img_width;
            b_t[3] = (center_y + box_height) / img_height;
            b_t += 4;
          }
          // priors with different aspect ratios
          for (float ar : new_aspect_ratios) {
            if (fabs(ar - 1.) < 1e-6) {
              continue;
            }
            box_width = min_size * sqrt(ar) / 2.;
            box_height = min_size / sqrt(ar) / 2.;
            b_t[0] = (center_x - box_width) / img_width;
            b_t[1] = (center_y - box_height) / img_height;
            b_t[2] = (center_x + box_width) / img_width;
            b_t[3] = (center_y + box_height) / img_height;
            b_t += 4;
          }
        } else {
          // priors with different aspect ratios
          for (auto ar : new_aspect_ratios) {
            box_width = min_size * sqrt(ar) / 2.;
            box_height = min_size / sqrt(ar) / 2.;
            b_t[0] = (center_x - box_width) / img_width;
            b_t[1] = (center_y - box_height) / img_height;
            b_t[2] = (center_x + box_width) / img_width;
            b_t[3] = (center_y + box_height) / img_height;
            b_t += 4;
          }
          if (!max_sizes.empty()) {
            auto max_size = max_sizes[s];
            // square prior with size sqrt(minSize * maxSize)
            box_width = box_height = sqrt(min_size * max_size) / 2.;
            b_t[0] = (center_x - box_width) / img_width;
            b_t[1] = (center_y - box_height) / img_height;
            b_t[2] = (center_x + box_width) / img_width;
            b_t[3] = (center_y + box_height) / img_height;
            b_t += 4;
          }
        }
      }
    }
  }

  if (clip) {
    T* dt = out->data<T>();
    std::transform(dt, dt + out->numel(), dt, [](T v) -> T {
      return std::min<T>(std::max<T>(v, 0.), 1.);
    });
  }

  DenseTensor var_t;
  var_t.Resize(common::make_ddim({1, static_cast<int>(variances.size())}));
  ctx.template Alloc<T>(&var_t);
  auto var_et = EigenTensor<T, 2>::From(var_t);

#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (size_t i = 0; i < variances.size(); ++i) {
    var_et(0, i) = variances[i];
  }

  int box_num = static_cast<int>(feature_height * feature_width * num_priors);
  auto var_dim = var->dims();
  var->Resize({box_num, static_cast<int>(variances.size())});

  auto e_vars = EigenMatrix<T, Eigen::RowMajor>::From(*var);

#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for collapse(2)
#endif
  for (int i = 0; i < box_num; ++i) {
    for (size_t j = 0; j < variances.size(); ++j) {
      e_vars(i, j) = variances[j];
    }
  }
  var->Resize(var_dim);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    prior_box, CPU, ALL_LAYOUT, phi::PriorBoxKernel, float, double) {}
