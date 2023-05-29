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

typedef struct PD_KernelKey PD_KernelKey;

typedef struct PD_Kernel PD_Kernel;

typedef struct PD_KernelArgsDef PD_KernelArgsDef;

typedef struct PD_TensorArgDef PD_TensorArgDef;

/**
 * TensorArgDef
 */

void PD_TensorArgDefSetDataLayout(PD_TensorArgDef *def,
                                  PD_DataLayout layout,
                                  PD_Status *status);

void PD_TensorArgDefSetDataType(PD_TensorArgDef *def,
                                PD_DataType dtype,
                                PD_Status *status);

/**
 * KernelArgsDef
 */

PD_List PD_KernelArgsDefGetInputArgDefs(PD_KernelArgsDef *def,
                                        PD_Status *status);

PD_List PD_KernelArgsDefGetOutputArgDefs(PD_KernelArgsDef *def,
                                         PD_Status *status);

/**
 * KernelKey
 */

PD_DataLayout PD_KernelKeyGetLayout(PD_KernelKey *key, PD_Status *status);

PD_DataType PD_KernelKeyGetDataType(PD_KernelKey *key, PD_Status *status);

/**
 * Kernel
 */

PD_KernelArgsDef *PD_KernelGetArgsDef(PD_Kernel *kernel, PD_Status *status);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
