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

#include "ModelConfig.pb.h"
#include "hl_gpu.h"
#include "paddle/utils/Common.h"

namespace paddle {

class GruCompute {
 public:
  void init(LayerConfig &config);

  template <bool useGpu>
  void forward(hl_gru_value value, int frameSize, int batchSize = 1);

  template <bool useGpu>
  void backward(hl_gru_value value,
                hl_gru_grad grad,
                int frameSize,
                int batchSize = 1);

 public:
  hl_activation_mode_t activeNode_;
  hl_activation_mode_t activeGate_;
};

}  // namespace paddle
