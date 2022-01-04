// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

///
/// \file pd_utils.h
///
/// \brief Some utility function to destroy paddle struct.
///
/// \author paddle-infer@baidu.com
/// \date 2021-04-21
/// \since 2.1
///

#pragma once

#include <stdint.h>
#include <stdio.h>

#include "pd_types.h"  // NOLINT

#ifdef __cplusplus
extern "C" {
#endif

///
/// \brief Destroy the PD_OneDimArrayInt32 object pointed to by the pointer.
///
/// \param[in] array pointer to the PD_OneDimArrayInt32 object.
///
PADDLE_CAPI_EXPORT extern void PD_OneDimArrayInt32Destroy(
    __pd_take PD_OneDimArrayInt32* array);

///
/// \brief Destroy the PD_OneDimArrayCstr object pointed to by the pointer.
///
/// \param[in] array pointer to the PD_OneDimArrayCstr object.
///
PADDLE_CAPI_EXPORT extern void PD_OneDimArrayCstrDestroy(
    __pd_take PD_OneDimArrayCstr* array);

///
/// \brief Destroy the PD_OneDimArraySize object pointed to by the pointer.
///
/// \param[in] array pointer to the PD_OneDimArraySize object.
///
PADDLE_CAPI_EXPORT extern void PD_OneDimArraySizeDestroy(
    __pd_take PD_OneDimArraySize* array);

///
/// \brief Destroy the PD_TwoDimArraySize object pointed to by the pointer.
///
/// \param[in] array pointer to the PD_TwoDimArraySize object.
///
PADDLE_CAPI_EXPORT extern void PD_TwoDimArraySizeDestroy(
    __pd_take PD_TwoDimArraySize* array);

///
/// \brief Destroy the PD_Cstr object pointed to by the pointer.
/// NOTE: if input string is empty, the return PD_Cstr's size is
/// 0 and data is NULL.
///
/// \param[in] cstr pointer to the PD_Cstr object.
///
PADDLE_CAPI_EXPORT extern void PD_CstrDestroy(__pd_take PD_Cstr* cstr);

#ifdef __cplusplus
}  // extern "C"
#endif
