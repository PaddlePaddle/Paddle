// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/lite/arm/math/saturate.h"
#include "paddle/fluid/lite/core/mir/pass.h"
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace mir {

/*
 * IoComplementPass complement the necessary instruction to make data
 * transferring or transformation between different places.
 */
class TransWeightPass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override;
  std::vector<float> GetWeightScale(float* in_data, int64_t axis_size,
                                    int64_t inner_size, float scale_factor) {
    std::vector<float> scale_out(axis_size);
    auto calc_abs_max = [&](float* in, size_t data_size) -> float {
      float max_data = 0.0;
      for (size_t i = 0; i < data_size; i++) {
        if (max_data < std::abs(in[i])) max_data = std::abs(in[i]);
      }
      return max_data;
    };
    for (int c = 0; c < axis_size; c++) {
      float* part_in = in_data + c * inner_size;
      scale_out[c] = calc_abs_max(part_in, inner_size) / scale_factor;
    }
    return scale_out;
  }
  void FP32ToInt8(const float* din, int8_t* dout, const float* scale,
                  int axis_size, int64_t outer_size, int64_t inner_size) {
    int loop_size = axis_size * outer_size;
    for (int i = 0; i < loop_size; ++i) {
      float inv_scale = 1.f / scale[i % axis_size];
      for (int j = 0; j < inner_size; ++j) {
        dout[j] = static_cast<int8_t>(std::roundf(din[j] * inv_scale));
      }
      dout += inner_size;
      din += inner_size;
    }
  }

  void TransFP32BiasToInt32(const float* din, int* dout, size_t data_size,
                            float in_scale, std::vector<float> weight_scale) {
    CHECK(data_size == weight_scale.size())
        << "Bias data size should be equal toe the weight scale data size.";
    for (size_t i = 0; i < data_size; i++) {
      dout[i] =
          static_cast<int>(std::roundf(din[i] / in_scale / weight_scale[i]));
    }
  }

  void SetValidPlaces(const std::vector<Place>& valid_places);

  const std::vector<Place>& valid_places() const { return valid_places_; }

 private:
  std::vector<Place> valid_places_;
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
