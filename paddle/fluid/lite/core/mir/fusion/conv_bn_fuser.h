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

#include <memory>
#include <string>
#include "paddle/fluid/lite/core/mir/pattern_matcher_high_api.h"

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

class ConvBNFuser : public FuseBase {
 public:
  explicit ConvBNFuser(const std::string& conv_type) : conv_type_(conv_type) {}
  void BuildPattern() override;
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override;

 private:
  cpp::OpDesc GenOpDesc(const key2nodes_t& matched) override;
  void ComputeFusedWeight(float* scale_d, float* mean_d, float* var_d,
                          float* bias_d, float* conv_weight_d, float eps, int h,
                          int w) {
    for (int i = 0; i < h; i++) {
      var_d[i] = scale_d[i] / std::sqrt(var_d[i] + eps);
    }
    for (int i = 0; i < h; i++) {
      bias_d[i] += (-mean_d[i]) * var_d[i];
    }
    for (int i = 0; i < h; i++) {
      for (int j = 0; j < w; j++) {
        conv_weight_d[i * w + j] *= var_d[i];
      }
    }
  }

 private:
  std::string conv_type_{"conv2d"};
};

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
