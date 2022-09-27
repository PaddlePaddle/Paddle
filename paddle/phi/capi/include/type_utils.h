// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#if !defined(_WIN32) && !defined(__APPLE__)

#include "paddle/phi/capi/include/c_data_type.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/layout.h"
#include "paddle/phi/core/enforce.h"

namespace phi {
namespace capi {

inline PD_DataType ToPDDataType(::paddle::experimental::DataType dtype) {
#define return_result(in, ret)               \
  case ::paddle::experimental::DataType::in: \
    return PD_DataType::ret
  switch (dtype) {
    return_result(UNDEFINED, UNDEFINED);
    return_result(FLOAT64, FLOAT64);
    return_result(FLOAT32, FLOAT32);
    return_result(FLOAT16, FLOAT16);
    return_result(BFLOAT16, BFLOAT16);
    return_result(INT64, INT64);
    return_result(INT32, INT32);
    return_result(INT16, INT16);
    return_result(INT8, INT8);
    return_result(UINT64, UINT64);
    return_result(UINT32, UINT32);
    return_result(UINT16, UINT16);
    return_result(UINT8, UINT8);
    return_result(BOOL, BOOL);
    default: {
      PADDLE_THROW(
          ::phi::errors::Unavailable("DataType %d is not supported.", dtype));
    }
  }
#undef return_result
}

inline ::paddle::experimental::DataType ToPhiDataType(PD_DataType dtype) {
#define return_result(in, ret) \
  case PD_DataType::in:        \
    return ::paddle::experimental::DataType::ret
  switch (dtype) {
    return_result(UNDEFINED, UNDEFINED);
    return_result(FLOAT64, FLOAT64);
    return_result(FLOAT32, FLOAT32);
    return_result(FLOAT16, FLOAT16);
    return_result(BFLOAT16, BFLOAT16);
    return_result(INT64, INT64);
    return_result(INT32, INT32);
    return_result(INT16, INT16);
    return_result(INT8, INT8);
    return_result(UINT64, UINT64);
    return_result(UINT32, UINT32);
    return_result(UINT16, UINT16);
    return_result(UINT8, UINT8);
    return_result(BOOL, BOOL);
    default: {
      PADDLE_THROW(
          ::phi::errors::Unavailable("DataType %d is not supported.", dtype));
      return ::paddle::experimental::DataType::UNDEFINED;
    }
  }
#undef return_result
}

inline PD_DataLayout ToPDDataLayout(::paddle::experimental::DataLayout layout) {
#define return_result(in, ret)                 \
  case ::paddle::experimental::DataLayout::in: \
    return PD_DataLayout::ret
  switch (layout) {
    return_result(ANY, ANY);
    return_result(NHWC, NHWC);
    return_result(NCHW, NCHW);
    return_result(NCDHW, NCDHW);
    return_result(NDHWC, NDHWC);
    default: {
      PADDLE_THROW(::phi::errors::Unavailable("DataLayout %d is not supported.",
                                              layout));
      return PD_DataLayout::ANY;
    }
  }
#undef return_result
}

inline ::paddle::experimental::DataLayout ToPhiDataLayout(
    PD_DataLayout layout) {
#define return_result(in, ret) \
  case PD_DataLayout::in:      \
    return ::paddle::experimental::DataLayout::ret
  switch (layout) {
    return_result(ANY, ANY);
    return_result(NHWC, NHWC);
    return_result(NCHW, NCHW);
    return_result(NCDHW, NCDHW);
    return_result(NDHWC, NDHWC);
    default: {
      PADDLE_THROW(::phi::errors::Unavailable("DataLayout %d is not supported.",
                                              layout));
      return ::paddle::experimental::DataLayout::ANY;
    }
  }
#undef return_result
}

}  // namespace capi
}  // namespace phi

#endif
