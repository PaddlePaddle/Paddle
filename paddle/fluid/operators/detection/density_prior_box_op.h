/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <algorithm>
#include <vector>
#include "paddle/fluid/operators/detection/prior_box_op.h"

namespace paddle {
namespace operators {

template <typename T>
class DensityPriorBoxOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<paddle::framework::Tensor>("Input");
    auto* image = ctx.Input<paddle::framework::Tensor>("Image");
    auto* boxes = ctx.Output<paddle::framework::Tensor>("Boxes");
    auto* vars = ctx.Output<paddle::framework::Tensor>("Variances");

    auto variances = ctx.Attr<std::vector<float>>("variances");
    auto clip = ctx.Attr<bool>("clip");

    auto fixed_sizes = ctx.Attr<std::vector<float>>("fixed_sizes");
    auto fixed_ratios = ctx.Attr<std::vector<float>>("fixed_ratios");
    auto densities = ctx.Attr<std::vector<int>>("densities");

    T step_w = static_cast<T>(ctx.Attr<float>("step_w"));
    T step_h = static_cast<T>(ctx.Attr<float>("step_h"));
    T offset = static_cast<T>(ctx.Attr<float>("offset"));

    auto img_width = image->dims()[3];
    auto img_height = image->dims()[2];

    auto feature_width = input->dims()[3];
    auto feature_height = input->dims()[2];

    T step_width, step_height;
    if (step_w == 0 || step_h == 0) {
      step_width = static_cast<T>(img_width) / feature_width;
      step_height = static_cast<T>(img_height) / feature_height;
    } else {
      step_width = step_w;
      step_height = step_h;
    }
    int num_priors = 0;

#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for reduction(+ : num_priors)
#endif
    for (size_t i = 0; i < densities.size(); ++i) {
      num_priors += (fixed_ratios.size()) * (pow(densities[i], 2));
    }

    boxes->mutable_data<T>(ctx.GetPlace());
    vars->mutable_data<T>(ctx.GetPlace());

    auto box_dim = vars->dims();
    boxes->Resize({feature_height, feature_width, num_priors, 4});
    auto e_boxes = framework::EigenTensor<T, 4>::From(*boxes).setConstant(0.0);
    int step_average = static_cast<int>((step_width + step_height) * 0.5);

    std::vector<float> sqrt_fixed_ratios;
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
    for (size_t i = 0; i < fixed_ratios.size(); i++) {
      sqrt_fixed_ratios.push_back(sqrt(fixed_ratios[i]));
    }

#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for collapse(2)
#endif
    for (int h = 0; h < feature_height; ++h) {
      for (int w = 0; w < feature_width; ++w) {
        T center_x = (w + offset) * step_width;
        T center_y = (h + offset) * step_height;
        int idx = 0;
        // Generate density prior boxes with fixed sizes.
        for (size_t s = 0; s < fixed_sizes.size(); ++s) {
          auto fixed_size = fixed_sizes[s];
          int density = densities[s];
          int shift = step_average / density;
          // Generate density prior boxes with fixed ratios.
          for (size_t r = 0; r < fixed_ratios.size(); ++r) {
            float box_width_ratio = fixed_size * sqrt_fixed_ratios[r];
            float box_height_ratio = fixed_size / sqrt_fixed_ratios[r];
            float density_center_x = center_x - step_average / 2. + shift / 2.;
            float density_center_y = center_y - step_average / 2. + shift / 2.;
            for (int di = 0; di < density; ++di) {
              for (int dj = 0; dj < density; ++dj) {
                float center_x_temp = density_center_x + dj * shift;
                float center_y_temp = density_center_y + di * shift;
                e_boxes(h, w, idx, 0) = std::max(
                    (center_x_temp - box_width_ratio / 2.) / img_width, 0.);
                e_boxes(h, w, idx, 1) = std::max(
                    (center_y_temp - box_height_ratio / 2.) / img_height, 0.);
                e_boxes(h, w, idx, 2) = std::min(
                    (center_x_temp + box_width_ratio / 2.) / img_width, 1.);
                e_boxes(h, w, idx, 3) = std::min(
                    (center_y_temp + box_height_ratio / 2.) / img_height, 1.);
                idx++;
              }
            }
          }
        }
      }
    }
    if (clip) {
      T* dt = boxes->data<T>();
      std::transform(dt, dt + boxes->numel(), dt, [](T v) -> T {
        return std::min<T>(std::max<T>(v, 0.), 1.);
      });
    }
    framework::Tensor var_t;
    var_t.mutable_data<T>(
        phi::make_ddim({1, static_cast<int>(variances.size())}),
        ctx.GetPlace());

    auto var_et = framework::EigenTensor<T, 2>::From(var_t);

    for (size_t i = 0; i < variances.size(); ++i) {
      var_et(0, i) = variances[i];
    }

    int box_num = feature_height * feature_width * num_priors;
    auto var_dim = vars->dims();
    vars->Resize({box_num, static_cast<int>(variances.size())});

    auto e_vars = framework::EigenMatrix<T, Eigen::RowMajor>::From(*vars);
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for collapse(2)
#endif
    for (int i = 0; i < box_num; ++i) {
      for (size_t j = 0; j < variances.size(); ++j) {
        e_vars(i, j) = variances[j];
      }
    }

    vars->Resize(var_dim);
    boxes->Resize(box_dim);
  }
};  // namespace operators

}  // namespace operators
}  // namespace paddle
