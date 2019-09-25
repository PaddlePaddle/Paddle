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
#include "paddle/fluid/platform/enforce.h"

template <PD_DataType DType>
struct GetDataType;
#define DECLARE_PD_DTYPE_CONVERTOR(PD_DATA_TYPE, REAL_TYPE) \
  template <>                                               \
  struct GetDataType<PD_DATA_TYPE> {                        \
    using RealType = REAL_TYPE;                             \
  }

DECLARE_PD_DTYPE_CONVERTOR(PD_FLOAT32, float);
DECLARE_PD_DTYPE_CONVERTOR(PD_INT32, int32_t);
DECLARE_PD_DTYPE_CONVERTOR(PD_INT64, int64_t);
DECLARE_PD_DTYPE_CONVERTOR(PD_UINT8, uint8_t);

#undef DECLARE_PD_DTYPE_CONVERTOR

using PD_PaddleDType = paddle::PaddleDType;
using PD_PaddlePlace = paddle::PaddlePlace;

struct PD_Predictor {
  std::unique_ptr<paddle::PaddlePredictor> predictor;
};

struct PD_AnalysisConfig {
  paddle::AnalysisConfig config;
} PD_AnalysisConfig;

struct InTensorShape {
  char* name;
  int* tensor_shape;
  int shape_size;
} InTensorShape;

struct PD_ZeroCopyTensor {
  paddle::ZeroCopyTensor tensor;
} PD_ZeroCopyTensor;

struct PD_Tensor {
  paddle::PaddleTensor tensor;
} PD_Tensor;

struct PD_PaddleBuf {
  paddle::PaddleBuf buf;
} PD_PaddleBuf;

struct PD_PaddleBuf {
  paddle::PaddleBuf buf;
} PD_PaddleBuf;

struct PD_MaxInputShape {
  char* name;
  int* shape;
  int shape_size;
} PD_MaxInputShape;

paddle::PaddleDType ConvertToPaddleDType(PD_DataType dtype) {
  switch (dtype) {
    case PD_FLOAT32:
      return PD_PaddleDType::Float32;
    case PD_INT32:
      return PD_PaddleDType::Int32;
    case PD_INT64:
      return PD_PaddleDType::INT64;
    case PD_UINT8:
      return PD_PaddleDType::UINT8;
  }
}
