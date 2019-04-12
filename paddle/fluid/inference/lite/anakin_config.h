// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <cassert>
#include <memory>
#include <string>
#include <vector>

#include "paddle_api.h"  // NOLINT
#include "utils/logger/logger.h"

namespace paddle {
namespace contrib {
// Configurations for Anakin engine.
struct AnakinConfig : public PaddlePredictor::Config {
  enum TargetType { ARM = 0, GPU };
  enum PrecisionType { FP32 = 0, FP16, INT8 };

  std::string model_file;
  int max_batch_size = 1;
  int thread_num = 1;
  TargetType target_type = ARM;
  PrecisionType precision_type = FP32;
};

}  // namespace contrib
}  // namespace paddle
