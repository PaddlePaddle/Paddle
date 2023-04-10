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

#if !defined(_WIN32)

#include "paddle/phi/capi/include/c_data_type.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PD_Scalar PD_Scalar;

bool PD_ScalarGetBoolData(PD_Scalar *scalar);

int8_t PD_ScalarGetInt8Data(PD_Scalar *scalar);

int16_t PD_ScalarGetInt16Data(PD_Scalar *scalar);

int32_t PD_ScalarGetInt32Data(PD_Scalar *scalar);

int64_t PD_ScalarGetInt64Data(PD_Scalar *scalar);

uint8_t PD_ScalarGetUInt8Data(PD_Scalar *scalar);

uint16_t PD_ScalarGetUInt16Data(PD_Scalar *scalar);

uint32_t PD_ScalarGetUInt32Data(PD_Scalar *scalar);

uint64_t PD_ScalarGetUInt64Data(PD_Scalar *scalar);

float PD_ScalarGetFloat32Data(PD_Scalar *scalar);

double PD_ScalarGetFloat64Data(PD_Scalar *scalar);

PD_DataType PD_ScalarGetDataType(PD_Scalar *scalar);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
