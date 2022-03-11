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

#include "paddle/infrt/dialect/phi/data_type.h"

namespace infrt {

phi::Backend cvtTarget2Phi(TargetType target) {
  switch (target) {
    case TargetType::CPU:
      return phi::Backend::CPU;
    case TargetType::GPU:
      return phi::Backend::GPU;
    default:
      return phi::Backend::UNDEFINED;
  }
}

TargetType cvtTargetFromPhi(phi::Backend backend) {
  switch (backend) {
    case phi::Backend::CPU:
      return TargetType::CPU;
    case phi::Backend::GPU:
      return TargetType::GPU;
    default:
      return TargetType::UNK;
  }
}

phi::DataType cvtPrecision2Phi(PrecisionType precision) {
#define CONVERT_PRECISION_TO_PHI(Precision) \
  case PrecisionType::Precision:            \
    return phi::DataType::Precision;

  switch (precision) {
    CONVERT_PRECISION_TO_PHI(FLOAT32)
    CONVERT_PRECISION_TO_PHI(FLOAT16)
    CONVERT_PRECISION_TO_PHI(FLOAT64)
    CONVERT_PRECISION_TO_PHI(UINT8)
    CONVERT_PRECISION_TO_PHI(INT8)
    CONVERT_PRECISION_TO_PHI(INT16)
    CONVERT_PRECISION_TO_PHI(INT32)
    CONVERT_PRECISION_TO_PHI(INT64)
    CONVERT_PRECISION_TO_PHI(COMPLEX64)
    CONVERT_PRECISION_TO_PHI(COMPLEX128)
    CONVERT_PRECISION_TO_PHI(BOOL)
    default:
      return phi::DataType::UNDEFINED;
  }
#undef CONVERT_PRECISION_TO_PHI
}

PrecisionType cvtPrecisionFromPhi(phi::DataType datatype) {
#define CONVERT_PRECISION_FROM_PHI(Precision) \
  case phi::DataType::Precision:              \
    return PrecisionType::Precision;

  switch (datatype) {
    CONVERT_PRECISION_FROM_PHI(FLOAT32)
    CONVERT_PRECISION_FROM_PHI(FLOAT16)
    CONVERT_PRECISION_FROM_PHI(FLOAT64)
    CONVERT_PRECISION_FROM_PHI(UINT8)
    CONVERT_PRECISION_FROM_PHI(INT8)
    CONVERT_PRECISION_FROM_PHI(INT16)
    CONVERT_PRECISION_FROM_PHI(INT32)
    CONVERT_PRECISION_FROM_PHI(INT64)
    CONVERT_PRECISION_FROM_PHI(COMPLEX64)
    CONVERT_PRECISION_FROM_PHI(COMPLEX128)
    CONVERT_PRECISION_FROM_PHI(BOOL)
    default:
      return PrecisionType::UNK;
  }
#undef CONVERT_PRECISION_FROM_PHI
}

phi::DataLayout cvtLayout2Phi(LayoutType layout) {
  switch (layout) {
    case LayoutType::NCHW:
      return phi::DataLayout::NCHW;
    case LayoutType::NHWC:
      return phi::DataLayout::NHWC;
    case LayoutType::ANY:
      return phi::DataLayout::ANY;
    default:
      return phi::DataLayout::UNDEFINED;
  }
}

LayoutType cvtLayoutFromPhi(phi::DataLayout layout) {
  switch (layout) {
    case phi::DataLayout::NCHW:
      return LayoutType::NCHW;
    case phi::DataLayout::NHWC:
      return LayoutType::NHWC;
    case phi::DataLayout::ANY:
      return LayoutType::ANY;
    default:
      return LayoutType::UNK;
  }
}

phi::KernelKey cvtPlace2Phi(const Place& place) {
  return phi::KernelKey(cvtTarget2Phi(place.target),
                        cvtLayout2Phi(place.layout),
                        cvtPrecision2Phi(place.precision));
}

Place cvtPlaceFromPhi(phi::TensorArgDef tensor_arg) {
  return Place(cvtTargetFromPhi(tensor_arg.backend),
               cvtPrecisionFromPhi(tensor_arg.dtype),
               cvtLayoutFromPhi(tensor_arg.layout));
}

}  // namespace infrt
