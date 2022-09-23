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
/// \file pd_tensor.h
///
/// \brief interface for paddle tensor
///
/// \author paddle-infer@baidu.com
/// \date 2021-04-21
/// \since 2.1
///

#pragma once

#include "pd_common.h"  // NOLINT

typedef struct PD_Tensor PD_Tensor;
typedef struct PD_OneDimArrayInt32 PD_OneDimArrayInt32;
typedef struct PD_TwoDimArraySize PD_TwoDimArraySize;

#ifdef __cplusplus
extern "C" {
#endif

///
/// \brief Destroy the paddle tensor
///
/// \param[in] pd_tensor tensor
///
PADDLE_CAPI_EXPORT extern void PD_TensorDestroy(__pd_take PD_Tensor* pd_tensor);

///
/// \brief Reset the shape of the tensor.
/// Generally it's only used for the input tensor.
/// Reshape must be called before calling PD_TensorMutableData*() or
/// PD_TensorCopyFromCpu*()
///
/// \param[in] pd_tensor tensor.
/// \param[in] shape_size The size of shape.
/// \param[in] shape The shape to set.
///
PADDLE_CAPI_EXPORT extern void PD_TensorReshape(__pd_keep PD_Tensor* pd_tensor,
                                                size_t shape_size,
                                                int32_t* shape);

///
/// \brief Get the memory pointer in CPU or GPU with 'float' data type.
/// Please Reshape the tensor first before call this.
/// It's usually used to get input data pointer.
///
/// \param[in] pd_tensor tensor.
/// \param[in] place The place of the tensor.
/// \return Memory pointer of pd_tensor
///
PADDLE_CAPI_EXPORT extern float* PD_TensorMutableDataFloat(
    __pd_keep PD_Tensor* pd_tensor, PD_PlaceType place);
///
/// \brief Get the memory pointer in CPU or GPU with 'int64_t' data type.
/// Please Reshape the tensor first before call this.
/// It's usually used to get input data pointer.
///
/// \param[in] pd_tensor tensor.
/// \param[in] place The place of the tensor.
/// \return Memory pointer of pd_tensor
///
PADDLE_CAPI_EXPORT extern int64_t* PD_TensorMutableDataInt64(
    __pd_keep PD_Tensor* pd_tensor, PD_PlaceType place);
///
/// \brief Get the memory pointer in CPU or GPU with 'int32_t' data type.
/// Please Reshape the tensor first before call this.
/// It's usually used to get input data pointer.
///
/// \param[in] pd_tensor tensor.
/// \param[in] place The place of the tensor.
/// \return Memory pointer of pd_tensor
///
PADDLE_CAPI_EXPORT extern int32_t* PD_TensorMutableDataInt32(
    __pd_keep PD_Tensor* pd_tensor, PD_PlaceType place);
///
/// \brief Get the memory pointer in CPU or GPU with 'uint8_t' data type.
/// Please Reshape the tensor first before call this.
/// It's usually used to get input data pointer.
///
/// \param[in] pd_tensor tensor.
/// \param[in] place The place of the tensor.
/// \return Memory pointer of pd_tensor
///
PADDLE_CAPI_EXPORT extern uint8_t* PD_TensorMutableDataUint8(
    __pd_keep PD_Tensor* pd_tensor, PD_PlaceType place);
///
/// \brief Get the memory pointer in CPU or GPU with 'int8_t' data type.
/// Please Reshape the tensor first before call this.
/// It's usually used to get input data pointer.
///
/// \param[in] pd_tensor tensor.
/// \param[in] place The place of the tensor.
/// \return Memory pointer of pd_tensor
///
PADDLE_CAPI_EXPORT extern int8_t* PD_TensorMutableDataInt8(
    __pd_keep PD_Tensor* pd_tensor, PD_PlaceType place);
///
/// \brief Get the memory pointer directly.
/// It's usually used to get the output data pointer.
///
/// \param[in] pd_tensor tensor.
/// \param[out] place To get the device type of the tensor.
/// \param[out] size To get the data size of the tensor.
/// \return The tensor data buffer pointer.
///
PADDLE_CAPI_EXPORT extern float* PD_TensorDataFloat(
    __pd_keep PD_Tensor* pd_tensor, PD_PlaceType* place, int32_t* size);
///
/// \brief Get the memory pointer directly.
/// It's usually used to get the output data pointer.
///
/// \param[in] pd_tensor tensor.
/// \param[out] place To get the device type of the tensor.
/// \param[out] size To get the data size of the tensor.
/// \return The tensor data buffer pointer.
///
PADDLE_CAPI_EXPORT extern int64_t* PD_TensorDataInt64(
    __pd_keep PD_Tensor* pd_tensor, PD_PlaceType* place, int32_t* size);
///
/// \brief Get the memory pointer directly.
/// It's usually used to get the output data pointer.
///
/// \param[in] pd_tensor tensor.
/// \param[out] place To get the device type of the tensor.
/// \param[out] size To get the data size of the tensor.
/// \return The tensor data buffer pointer.
///
PADDLE_CAPI_EXPORT extern int32_t* PD_TensorDataInt32(
    __pd_keep PD_Tensor* pd_tensor, PD_PlaceType* place, int32_t* size);
///
/// \brief Get the memory pointer directly.
/// It's usually used to get the output data pointer.
///
/// \param[in] pd_tensor tensor.
/// \param[out] place To get the device type of the tensor.
/// \param[out] size To get the data size of the tensor.
/// \return The tensor data buffer pointer.
///
PADDLE_CAPI_EXPORT extern uint8_t* PD_TensorDataUint8(
    __pd_keep PD_Tensor* pd_tensor, PD_PlaceType* place, int32_t* size);
///
/// \brief Get the memory pointer directly.
/// It's usually used to get the output data pointer.
///
/// \param[in] pd_tensor tensor.
/// \param[out] place To get the device type of the tensor.
/// \param[out] size To get the data size of the tensor.
/// \return The tensor data buffer pointer.
///
PADDLE_CAPI_EXPORT extern int8_t* PD_TensorDataInt8(
    __pd_keep PD_Tensor* pd_tensor, PD_PlaceType* place, int32_t* size);
///
/// \brief Copy the host memory to tensor data.
/// It's usually used to set the input tensor data.
/// \param[in] pd_tensor tensor.
/// \param[in] data The pointer of the data, from which the tensor will copy.
///
PADDLE_CAPI_EXPORT extern void PD_TensorCopyFromCpuFloat(
    __pd_keep PD_Tensor* pd_tensor, const float* data);
///
/// \brief Copy the host memory to tensor data.
/// It's usually used to set the input tensor data.
/// \param[in] pd_tensor tensor.
/// \param[in] data The pointer of the data, from which the tensor will copy.
///
PADDLE_CAPI_EXPORT extern void PD_TensorCopyFromCpuInt64(
    __pd_keep PD_Tensor* pd_tensor, const int64_t* data);
///
/// \brief Copy the host memory to tensor data.
/// It's usually used to set the input tensor data.
/// \param[in] pd_tensor tensor.
/// \param[in] data The pointer of the data, from which the tensor will copy.
///
PADDLE_CAPI_EXPORT extern void PD_TensorCopyFromCpuInt32(
    __pd_keep PD_Tensor* pd_tensor, const int32_t* data);
///
/// \brief Copy the host memory to tensor data.
/// It's usually used to set the input tensor data.
/// \param[in] pd_tensor tensor.
/// \param[in] data The pointer of the data, from which the tensor will copy.
///
PADDLE_CAPI_EXPORT extern void PD_TensorCopyFromCpuUint8(
    __pd_keep PD_Tensor* pd_tensor, const uint8_t* data);
///
/// \brief Copy the host memory to tensor data.
/// It's usually used to set the input tensor data.
/// \param[in] pd_tensor tensor.
/// \param[in] data The pointer of the data, from which the tensor will copy.
///
PADDLE_CAPI_EXPORT extern void PD_TensorCopyFromCpuInt8(
    __pd_keep PD_Tensor* pd_tensor, const int8_t* data);
///
/// \brief Copy the tensor data to the host memory.
/// It's usually used to get the output tensor data.
/// \param[in] pd_tensor tensor.
/// \param[out] data The tensor will copy the data to the address.
///
PADDLE_CAPI_EXPORT extern void PD_TensorCopyToCpuFloat(
    __pd_keep PD_Tensor* pd_tensor, float* data);
///
/// \brief Copy the tensor data to the host memory.
/// It's usually used to get the output tensor data.
/// \param[in] pd_tensor tensor.
/// \param[out] data The tensor will copy the data to the address.
///
PADDLE_CAPI_EXPORT extern void PD_TensorCopyToCpuInt64(
    __pd_keep PD_Tensor* pd_tensor, int64_t* data);
///
/// \brief Copy the tensor data to the host memory.
/// It's usually used to get the output tensor data.
/// \param[in] pd_tensor tensor.
/// \param[out] data The tensor will copy the data to the address.
///
PADDLE_CAPI_EXPORT extern void PD_TensorCopyToCpuInt32(
    __pd_keep PD_Tensor* pd_tensor, int32_t* data);
///
/// \brief Copy the tensor data to the host memory.
/// It's usually used to get the output tensor data.
/// \param[in] pd_tensor tensor.
/// \param[out] data The tensor will copy the data to the address.
///
PADDLE_CAPI_EXPORT extern void PD_TensorCopyToCpuUint8(
    __pd_keep PD_Tensor* pd_tensor, uint8_t* data);
///
/// \brief Copy the tensor data to the host memory.
/// It's usually used to get the output tensor data.
/// \param[in] pd_tensor tensor.
/// \param[out] data The tensor will copy the data to the address.
///
PADDLE_CAPI_EXPORT extern void PD_TensorCopyToCpuInt8(
    __pd_keep PD_Tensor* pd_tensor, int8_t* data);
///
/// \brief Get the tensor shape
/// \param[in] pd_tensor tensor.
/// \return The tensor shape.
///
PADDLE_CAPI_EXPORT extern __pd_give PD_OneDimArrayInt32* PD_TensorGetShape(
    __pd_keep PD_Tensor* pd_tensor);

///
/// \brief Set the tensor lod information
/// \param[in] pd_tensor tensor.
/// \param[in] lod lod information.
///
PADDLE_CAPI_EXPORT extern void PD_TensorSetLod(
    __pd_keep PD_Tensor* pd_tensor, __pd_keep PD_TwoDimArraySize* lod);
///
/// \brief Get the tensor lod information
/// \param[in] pd_tensor tensor.
/// \return the lod information.
///
PADDLE_CAPI_EXPORT extern __pd_give PD_TwoDimArraySize* PD_TensorGetLod(
    __pd_keep PD_Tensor* pd_tensor);
///
/// \brief Get the tensor name
/// \param[in] pd_tensor tensor.
/// \return the tensor name.
///
PADDLE_CAPI_EXPORT extern const char* PD_TensorGetName(
    __pd_keep PD_Tensor* pd_tensor);
///
/// \brief Get the tensor data type
/// \param[in] pd_tensor tensor.
/// \return the tensor data type.
///
PADDLE_CAPI_EXPORT extern PD_DataType PD_TensorGetDataType(
    __pd_keep PD_Tensor* pd_tensor);

#ifdef __cplusplus
}  // extern "C"
#endif
