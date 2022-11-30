/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/transform.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

inline void ExpandAspectRatios(const std::vector<float>& input_aspect_ratior,
                               bool flip,
                               std::vector<float>* output_aspect_ratior) {
  constexpr float epsilon = 1e-6;
  output_aspect_ratior->clear();
  output_aspect_ratior->push_back(1.0f);
  for (size_t i = 0; i < input_aspect_ratior.size(); ++i) {
    float ar = input_aspect_ratior[i];
    bool already_exist = false;
    for (size_t j = 0; j < output_aspect_ratior->size(); ++j) {
      if (fabs(ar - output_aspect_ratior->at(j)) < epsilon) {
        already_exist = true;
        break;
      }
    }
    if (!already_exist) {
      output_aspect_ratior->push_back(ar);
      if (flip) {
        output_aspect_ratior->push_back(1.0f / ar);
      }
    }
  }
}

template <typename T>
class PriorBoxOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* image = ctx.Input<phi::DenseTensor>("Image");

    PD_VISIT_FLOATING_TYPES(image->dtype(), "PriorBoxOpHandler", ([&] {
                              PriorBoxOpHandler<data_t>(ctx);
                            }));
  }

  template <typename K>
  void PriorBoxOpHandler(const framework::ExecutionContext& ctx) const {
    auto* input = ctx.Input<phi::DenseTensor>("Input");
    auto* image = ctx.Input<phi::DenseTensor>("Image");
    auto* boxes = ctx.Output<phi::DenseTensor>("Boxes");
    auto* vars = ctx.Output<phi::DenseTensor>("Variances");

    auto min_sizes = ctx.Attr<std::vector<float>>("min_sizes");
    auto max_sizes = ctx.Attr<std::vector<float>>("max_sizes");
    auto input_aspect_ratio = ctx.Attr<std::vector<float>>("aspect_ratios");
    auto variances = ctx.Attr<std::vector<float>>("variances");
    auto flip = ctx.Attr<bool>("flip");
    auto clip = ctx.Attr<bool>("clip");
    auto min_max_aspect_ratios_order =
        ctx.Attr<bool>("min_max_aspect_ratios_order");

    std::vector<float> aspect_ratios;
    ExpandAspectRatios(input_aspect_ratio, flip, &aspect_ratios);

    K step_w = static_cast<K>(ctx.Attr<float>("step_w"));
    K step_h = static_cast<K>(ctx.Attr<float>("step_h"));
    K offset = static_cast<K>(ctx.Attr<float>("offset"));

    auto img_width = image->dims()[3];
    auto img_height = image->dims()[2];

    auto feature_width = input->dims()[3];
    auto feature_height = input->dims()[2];

    K step_width, step_height;
    if (step_w == 0 || step_h == 0) {
      step_width = static_cast<K>(img_width) / feature_width;
      step_height = static_cast<K>(img_height) / feature_height;
    } else {
      step_width = step_w;
      step_height = step_h;
    }

    int num_priors = aspect_ratios.size() * min_sizes.size();
    if (max_sizes.size() > 0) {
      num_priors += max_sizes.size();
    }

    boxes->mutable_data<K>(ctx.GetPlace());
    vars->mutable_data<K>(ctx.GetPlace());

    K* b_t = boxes->data<K>();
    for (int h = 0; h < feature_height; ++h) {
      for (int w = 0; w < feature_width; ++w) {
        K center_x = (w + offset) * step_width;
        K center_y = (h + offset) * step_height;
        K box_width, box_height;
        for (size_t s = 0; s < min_sizes.size(); ++s) {
          auto min_size = min_sizes[s];
          if (min_max_aspect_ratios_order) {
            box_width = box_height = min_size / 2.;
            b_t[0] = (center_x - box_width) / img_width;
            b_t[1] = (center_y - box_height) / img_height;
            b_t[2] = (center_x + box_width) / img_width;
            b_t[3] = (center_y + box_height) / img_height;
            b_t += 4;
            if (max_sizes.size() > 0) {
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
            for (size_t r = 0; r < aspect_ratios.size(); ++r) {
              float ar = aspect_ratios[r];
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
            for (size_t r = 0; r < aspect_ratios.size(); ++r) {
              float ar = aspect_ratios[r];
              box_width = min_size * sqrt(ar) / 2.;
              box_height = min_size / sqrt(ar) / 2.;
              b_t[0] = (center_x - box_width) / img_width;
              b_t[1] = (center_y - box_height) / img_height;
              b_t[2] = (center_x + box_width) / img_width;
              b_t[3] = (center_y + box_height) / img_height;
              b_t += 4;
            }
            if (max_sizes.size() > 0) {
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
      K* dt = boxes->data<K>();
      std::transform(dt, dt + boxes->numel(), dt, [](K v) -> K {
        return std::min<K>(std::max<K>(v, 0.), 1.);
      });
    }

    phi::DenseTensor var_t;
    var_t.mutable_data<K>(
        phi::make_ddim({1, static_cast<int>(variances.size())}),
        ctx.GetPlace());
    auto var_et = phi::EigenTensor<K, 2>::From(var_t);

#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
    for (size_t i = 0; i < variances.size(); ++i) {
      var_et(0, i) = variances[i];
    }

    int box_num = feature_height * feature_width * num_priors;
    auto var_dim = vars->dims();
    vars->Resize({box_num, static_cast<int>(variances.size())});

    auto e_vars = phi::EigenMatrix<K, Eigen::RowMajor>::From(*vars);

#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for collapse(2)
#endif
    for (int i = 0; i < box_num; ++i) {
      for (size_t j = 0; j < variances.size(); ++j) {
        e_vars(i, j) = variances[j];
      }
    }
    vars->Resize(var_dim);
  }
};

}  // namespace operators
}  // namespace paddle
