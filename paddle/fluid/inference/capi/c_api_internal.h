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
#include "paddle/fluid/inference/api/paddle_analysis_config.h"
#include "paddle/fluid/inference/api/paddle_api.h"
#include "paddle/fluid/inference/capi/paddle_c_api.h"
#include "paddle/fluid/platform/enforce.h"

using PD_PaddleDType = paddle::PaddleDType;
using PD_ACPrecision = paddle::AnalysisConfig::Precision;

struct PD_AnalysisConfig {
  paddle::AnalysisConfig config;
};

struct PD_Tensor {
  paddle::PaddleTensor tensor;
};

struct PD_PaddleBuf {
  paddle::PaddleBuf buf;
};

struct PD_Predictor {
  std::unique_ptr<paddle::PaddlePredictor> predictor;
};

namespace paddle {
paddle::PaddleDType ConvertToPaddleDType(PD_DataType dtype);

PD_DataType ConvertToPDDataType(PD_PaddleDType dtype);

PD_ACPrecision ConvertToACPrecision(Precision dtype);
}  // namespace paddle
