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
/// \file utils_internal.h
///
/// \brief Some utility function used to convert object between C Struct and C++
/// Class.
///
/// \author paddle-infer@baidu.com
/// \date 2021-04-21
/// \since 2.1
///

#pragma once

#include <cstdint>
#include <cstdio>
#include <vector>

#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/inference/capi_exp/pd_types.h"

namespace paddle_infer {

///
/// \brief Convert the 'std::vector<int>' object to a 'PD_OneDimArrayInt32'
/// object.
///
/// \param[in] vec source object.
/// \return target object.
///
__pd_give PD_OneDimArrayInt32* CvtVecToOneDimArrayInt32(
    const std::vector<int>& vec);

///
/// \brief Convert the 'PD_OneDimArrayInt32' object to a 'std::vector<int>'
/// object.
///
/// \param[in] array source object.
/// \return target object.
///
std::vector<int> CvtOneDimArrayToVecInt32(
    __pd_keep const PD_OneDimArrayInt32* array);

///
/// \brief Convert the 'std::vector<size_t>' object to a 'PD_OneDimArraySize'
/// object.
///
/// \param[in] vec source object.
/// \return target object.
///
__pd_give PD_OneDimArraySize* CvtVecToOneDimArraySize(
    const std::vector<size_t>& vec);

///
/// \brief Convert the 'PD_OneDimArraySize' object to a 'std::vector<size_t>'
/// object.
///
/// \param[in] array source object.
/// \return target object.
///
std::vector<size_t> CvtOneDimArrayToVecSize(
    __pd_keep const PD_OneDimArraySize* array);

///
/// \brief Convert the 'std::vector<std::string>' object to a
/// 'PD_OneDimArrayCstr' object.
///
/// \param[in] vec source object.
/// \return target object.
///
__pd_give PD_OneDimArrayCstr* CvtVecToOneDimArrayCstr(
    const std::vector<std::string>& vec);

///
/// \brief Convert the 'PD_OneDimArrayCstr' object to a
/// 'std::vector<std::string>' object.
///
/// \param[in] array source object.
/// \return target object.
///
std::vector<std::string> CvtOneDimArrayToVecCstr(
    __pd_keep const PD_OneDimArrayCstr* array);

///
/// \brief Convert the 'std::vector<std::vector<size_t>>' object to a
/// 'PD_TwoDimArraySize' object.
///
/// \param[in] vec source object.
/// \return target object.
///
__pd_give PD_TwoDimArraySize* CvtVecToTwoDimArraySize(
    const std::vector<std::vector<size_t>>& vec);

///
/// \brief Convert the 'PD_TwoDimArraySize' object to a
/// 'std::vector<std::vector<size_t>>' object.
///
/// \param[in] array source object.
/// \return target object.
///
std::vector<std::vector<size_t>> CvtTwoDimArrayToVecSize(
    __pd_keep const PD_TwoDimArraySize* array);

///
/// \brief Convert the 'std::string' object to a 'PD_Cstr' object.
///
/// \param[in] vec source object.
/// \return target object.
///
__pd_give PD_Cstr* CvtStrToCstr(const std::string& vec);

///
/// \brief Convert the 'PD_PlaceType' object to a 'paddle_infer::PlaceType'
/// object.
///
/// \param[in] place_type source object.
/// \return target object.
///
PlaceType CvtToCxxPlaceType(PD_PlaceType place_type);

///
/// \brief Convert the 'paddle_infer::PlaceType' object to a 'PD_PlaceType'
/// object.
///
/// \param[in] place_type source object.
/// \return target object.
///
PD_PlaceType CvtFromCxxPlaceType(PlaceType place_type);

///
/// \brief Convert the 'PD_DataType' object to a 'paddle_infer::DataType'
/// object.
///
/// \param[in] place_type source object.
/// \return target object.
///
DataType CvtToCxxDatatype(PD_DataType data_type);

///
/// \brief Convert the 'paddle_infer::DataType' object to a 'PD_DataType'
/// object.
///
/// \param[in] place_type source object.
/// \return target object.
///
PD_DataType CvtFromCxxDatatype(DataType data_type);

}  // namespace paddle_infer
