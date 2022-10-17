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
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

constexpr int kPriorBoxFLOAT = 1;
constexpr int kPriorBoxDOUBLE = 2;

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

#define PD_VISIT_FLOAT_AND_DOUBLE_TYPES(TYPE, NAME, ...)                  \
  [&] {                                                                   \
    const auto& __dtype__ = TYPE;                                         \
    switch (__dtype__) {                                                  \
      PD_PRIVATE_CASE_TYPE(                                               \
          NAME, ::paddle::DataType::FLOAT32, float, __VA_ARGS__)          \
      PD_PRIVATE_CASE_TYPE(                                               \
          NAME, ::paddle::DataType::FLOAT64, double, __VA_ARGS__)         \
      default:                                                            \
        PD_THROW("function " #NAME " is not implemented for data type `", \
                 __dtype__,                                               \
                 "`");                                                    \
    }                                                                     \
  }()

template <typename T>
class PriorBoxOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
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

    PD_VISIT_FLOAT_AND_DOUBLE_TYPES(
        image->dtype(), "PriorBoxOpKernel::Compute", ([&] {
          std::vector<float> aspect_ratios;
          ExpandAspectRatios(input_aspect_ratio, flip, &aspect_ratios);

          data_t step_w = static_cast<data_t>(ctx.Attr<float>("step_w"));
          data_t step_h = static_cast<data_t>(ctx.Attr<float>("step_h"));
          data_t offset = static_cast<data_t>(ctx.Attr<float>("offset"));

          auto img_width = image->dims()[3];
          auto img_height = image->dims()[2];

          auto feature_width = input->dims()[3];
          auto feature_height = input->dims()[2];

          data_t step_width, step_height;
          if (step_w == 0 || step_h == 0) {
            step_width = static_cast<data_t>(img_width) / feature_width;
            step_height = static_cast<data_t>(img_height) / feature_height;
          } else {
            step_width = step_w;
            step_height = step_h;
          }

          int num_priors = aspect_ratios.size() * min_sizes.size();
          if (max_sizes.size() > 0) {
            num_priors += max_sizes.size();
          }

          boxes->mutable_data<data_t>(ctx.GetPlace());
          vars->mutable_data<data_t>(ctx.GetPlace());

          data_t* b_t = boxes->data<data_t>();
          for (int h = 0; h < feature_height; ++h) {
            for (int w = 0; w < feature_width; ++w) {
              data_t center_x = (w + offset) * step_width;
              data_t center_y = (h + offset) * step_height;
              data_t box_width, box_height;
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
            data_t* dt = boxes->data<data_t>();
            std::transform(dt, dt + boxes->numel(), dt, [](data_t v) -> data_t {
              return std::min<data_t>(std::max<data_t>(v, 0.), 1.);
            });
          }

          phi::DenseTensor var_t;
          var_t.mutable_data<data_t>(
              phi::make_ddim({1, static_cast<int>(variances.size())}),
              ctx.GetPlace());
          auto var_et = framework::EigenTensor<data_t, 2>::From(var_t);

#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
          for (size_t i = 0; i < variances.size(); ++i) {
            var_et(0, i) = variances[i];
          }

          int box_num = feature_height * feature_width * num_priors;
          auto var_dim = vars->dims();
          vars->Resize({box_num, static_cast<int>(variances.size())});

          auto e_vars =
              framework::EigenMatrix<data_t, Eigen::RowMajor>::From(*vars);

#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for collapse(2)
#endif
          for (int i = 0; i < box_num; ++i) {
            for (size_t j = 0; j < variances.size(); ++j) {
              e_vars(i, j) = variances[j];
            }
          }
          vars->Resize(var_dim);
        }));
  }
};  // namespace operators

}  // namespace operators
}  // namespace paddle
