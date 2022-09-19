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

#include "paddle/phi/capi/include/c_scalar.h"

#include "paddle/phi/capi/include/common.h"
#include "paddle/phi/capi/include/type_utils.h"
#include "paddle/phi/common/scalar.h"

PD_DataType PD_ScalarGetType(PD_Scalar* scalar) {
  auto cc_scalar = reinterpret_cast<phi::Scalar*>(scalar);
  return phi::capi::ToPDDataType(cc_scalar->dtype());
}

bool PD_ScalarGetBoolData(PD_Scalar* scalar) {
  auto cc_scalar = reinterpret_cast<phi::Scalar*>(scalar);
  return cc_scalar->to<bool>();
}

int8_t PD_ScalarGetInt8Data(PD_Scalar* scalar) {
  auto cc_scalar = reinterpret_cast<phi::Scalar*>(scalar);
  return cc_scalar->to<int8_t>();
}

int16_t PD_ScalarGetInt16Data(PD_Scalar* scalar) {
  auto cc_scalar = reinterpret_cast<phi::Scalar*>(scalar);
  return cc_scalar->to<int16_t>();
}

int32_t PD_ScalarGetInt32Data(PD_Scalar* scalar) {
  auto cc_scalar = reinterpret_cast<phi::Scalar*>(scalar);
  return cc_scalar->to<int32_t>();
}

int64_t PD_ScalarGetInt64Data(PD_Scalar* scalar) {
  auto cc_scalar = reinterpret_cast<phi::Scalar*>(scalar);
  return cc_scalar->to<int64_t>();
}

uint8_t PD_ScalarGetUInt8Data(PD_Scalar* scalar) {
  auto cc_scalar = reinterpret_cast<phi::Scalar*>(scalar);
  return cc_scalar->to<uint8_t>();
}

uint16_t PD_ScalarGetUInt16Data(PD_Scalar* scalar) {
  auto cc_scalar = reinterpret_cast<phi::Scalar*>(scalar);
  return cc_scalar->to<uint16_t>();
}

uint32_t PD_ScalarGetUInt32Data(PD_Scalar* scalar) {
  auto cc_scalar = reinterpret_cast<phi::Scalar*>(scalar);
  return cc_scalar->to<uint32_t>();
}

uint64_t PD_ScalarGetUInt64Data(PD_Scalar* scalar) {
  auto cc_scalar = reinterpret_cast<phi::Scalar*>(scalar);
  return cc_scalar->to<uint64_t>();
}

float PD_ScalarGetFloat32Data(PD_Scalar* scalar) {
  auto cc_scalar = reinterpret_cast<phi::Scalar*>(scalar);
  return cc_scalar->to<float>();
}

double PD_ScalarGetFloat64Data(PD_Scalar* scalar) {
  auto cc_scalar = reinterpret_cast<phi::Scalar*>(scalar);
  return cc_scalar->to<double>();
}

phi::dtype::float16 PD_ScalarGetFloat16Data(PD_Scalar* scalar) {
  auto cc_scalar = reinterpret_cast<phi::Scalar*>(scalar);
  return cc_scalar->to<phi::dtype::float16>();
}

PD_REGISTER_CAPI(scalar);
