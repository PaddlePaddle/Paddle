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

struct PD_Predictor {
  std::unique_ptr<paddle::PaddlePredictor> predictor;
};

typedef struct PD_AnalysisConfig {
  paddle::AnalysisConfig config;
} PD_AnalysisConfig;

typedef struct InTensorShape {
  char* name;
  int* tensor_shape;
  int shape_size;
} InTensorShape;

typedef struct PD_ZeroCopyTensor {
  paddle::ZeroCopyTensor tensor;
} PD_ZeroCopyTensor;

typedef struct PD_Tensor { paddle::PaddleTensor tensor; } PD_Tensor;

typedef struct PD_PaddleBuf { paddle::PaddleBuf buf; } PD_PaddleBuf;

typedef struct PD_PaddleBuf { paddle::PaddleBuf buf; } PD_PaddleBuf;

typedef struct PD_MaxInputShape {
  char* name;
  int* shape;
  int shape_size;
} PD_MaxInputShape;
