/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
#include "paddle/framework/op_registry.h"
#include "paddle/operators/math/math_function.h"
// #include "paddle/operators/strided_memcpy.h"

namespace paddle {
namespace operators {

inline void expand_aspect_ratios(const std::vector<float> input_aspect_ratior,
                                 bool flip,
                                 std::vector<float>& output_aspect_ratior) {
  constexpr float eps = 1e-6;
  output_aspect_ratior.clear();
  output_aspect_ratior.push_back(1.);
  for (size_t i = 0; i < input_aspect_ratior.size(); ++i) {
    float ar = input_aspect_ratior[i];
    bool already_exist = false;
    for (size_t j = 0; j < output_aspect_ratior.size(); ++j) {
      if (fabs(ar - output_aspect_ratior[j]) < eps) {
        already_exist = true;
        break;
      }
    }
    if (!already_exist) {
      output_aspect_ratior.push_back(ar);
      if (flip) {
        output_aspect_ratior.push_back(1. / ar);
      }
    }
  }
}

template <typename Place, typename T>
class PriorBoxOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<paddle::framework::Tensor>("Input");
    auto* image = ctx.Input<paddle::framework::Tensor>("Image");
    auto* out = ctx.Output<paddle::framework::Tensor>("Out");

    auto min_sizes = ctx.Attr<std::vector<int>>("min_sizes");
    auto max_sizes = ctx.Attr<std::vector<int>>("max_sizes");
    auto input_aspect_ratio = ctx.Attr<std::vector<float>>("aspect_ratios");
    auto variances = ctx.Attr<std::vector<float>>("variances");
    auto flip = ctx.Attr<bool>("flip");
    auto clip = ctx.Attr<bool>("clip");

    std::vector<float> aspect_ratios;
    expand_aspect_ratios(input_aspect_ratio, flip, aspect_ratios);

    auto img_w = ctx.Attr<int>("img_w");
    auto img_h = ctx.Attr<int>("img_h");
    auto step_w = ctx.Attr<float>("step_w");
    auto step_h = ctx.Attr<float>("step_h");
    auto offset = ctx.Attr<float>("offset");

    int img_width, img_height;
    if (img_h == 0 || img_w == 0) {
      img_width = image->dims()[3];
      img_height = image->dims()[2];
    } else {
      img_width = img_w;
      img_height = img_h;
    }

    const int layer_width = input->dims()[3];
    const int layer_height = input->dims()[2];

    float step_width, step_height;
    if (step_w == 0 || step_h == 0) {
      step_width = static_cast<float>(img_width) / layer_width;
      step_height = static_cast<float>(img_height) / layer_height;
    } else {
      step_width = step_w;
      step_height = step_h;
    }

    int num_priors = aspect_ratios.size() * min_sizes.size();
    if (max_sizes.size() > 0) {
      num_priors += max_sizes.size();
    }

    int dim = layer_height * layer_width * num_priors * 4;

    T* output_data = nullptr;
    framework::Tensor output_cpu;
    out->mutable_data<T>(ctx.GetPlace());
    if (platform::is_gpu_place(ctx.GetPlace())) {
      output_data =
          output_cpu.mutable_data<T>(out->dims(), platform::CPUPlace());
    } else {
      output_data = out->mutable_data<T>(ctx.GetPlace());
    }

    int idx = 0;
    for (int h = 0; h < layer_height; ++h) {
      for (int w = 0; w < layer_width; ++w) {
        float center_x = (w + offset) * step_width;
        float center_y = (h + offset) * step_height;
        float box_width, box_height;
        for (size_t s = 0; s < min_sizes.size(); ++s) {
          int min_size = min_sizes[s];
          // first prior: aspect_ratio = 1, size = min_size
          box_width = box_height = min_size;
          // xmin
          output_data[idx++] = (center_x - box_width / 2.) / img_width;
          // ymin
          output_data[idx++] = (center_y - box_height / 2.) / img_height;
          // xmax
          output_data[idx++] = (center_x + box_width / 2.) / img_width;
          // ymax
          output_data[idx++] = (center_y + box_height / 2.) / img_height;

          if (max_sizes.size() > 0) {
            int max_size = max_sizes[s];
            // second prior: aspect_ratio = 1,
            // size = sqrt(min_size * max_size)
            box_width = box_height = sqrt(min_size * max_size);
            // xmin
            output_data[idx++] = (center_x - box_width / 2.) / img_width;
            // ymin
            output_data[idx++] = (center_y - box_height / 2.) / img_height;
            // xmax
            output_data[idx++] = (center_x + box_width / 2.) / img_width;
            // ymax
            output_data[idx++] = (center_y + box_height / 2.) / img_height;
          }

          // rest of priors
          for (size_t r = 0; r < aspect_ratios.size(); ++r) {
            float ar = aspect_ratios[r];
            if (fabs(ar - 1.) < 1e-6) {
              continue;
            }
            box_width = min_size * sqrt(ar);
            box_height = min_size / sqrt(ar);
            // xmin
            output_data[idx++] = (center_x - box_width / 2.) / img_width;
            // ymin
            output_data[idx++] = (center_y - box_height / 2.) / img_height;
            // xmax
            output_data[idx++] = (center_x + box_width / 2.) / img_width;
            // ymax
            output_data[idx++] = (center_y + box_height / 2.) / img_height;
          }
        }
      }
    }

    // clip the prior's coordidate such that it is within [0, 1]
    if (clip) {
      for (int d = 0; d < dim; ++d) {
        output_data[d] = std::min<T>(std::max<T>(output_data[d], 0.), 1.);
      }
    }

    // set the variance.
    auto output_stride = framework::stride(out->dims());
    output_data += output_stride[1];
    if (variances.size() == 1) {
      for (int i = 0; i < dim; ++i) {
        output_data[i] = variances[0];
      }
    } else {
      int count = 0;
      for (int h = 0; h < layer_height; ++h) {
        for (int w = 0; w < layer_width; ++w) {
          for (int i = 0; i < num_priors; ++i) {
            for (int j = 0; j < 4; ++j) {
              output_data[count] = variances[j];
              ++count;
            }
          }
        }
      }
    }
    if (platform::is_gpu_place(ctx.GetPlace())) {
      framework::CopyFrom(output_cpu, platform::CPUPlace(),
                          ctx.device_context(), out);
    }
  }
};

}  // namespace operators
}  // namespace paddle
