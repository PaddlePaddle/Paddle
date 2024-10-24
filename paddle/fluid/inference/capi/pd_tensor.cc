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

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "paddle/fluid/inference/capi/c_api_internal.h"
#include "paddle/fluid/inference/capi/paddle_c_api.h"
#include "paddle/fluid/platform/enforce.h"

using paddle::ConvertToACPrecision;
using paddle::ConvertToPaddleDType;
using paddle::ConvertToPDDataType;

extern "C" {
// PaddleTensor
PD_Tensor* PD_NewPaddleTensor() { return new PD_Tensor; }

void PD_DeletePaddleTensor(PD_Tensor* tensor) {
  if (tensor) {
    delete tensor;
    tensor = nullptr;
    VLOG(3) << "PD_Tensor delete successfully. ";
  }
}

void PD_SetPaddleTensorName(PD_Tensor* tensor, char* name) {
  PADDLE_ENFORCE_NOT_NULL(tensor,
                          common::errors::InvalidArgument(
                              "The pointer of tensor shouldn't be nullptr"));
  tensor->tensor.name = std::string(name);
}

void PD_SetPaddleTensorDType(PD_Tensor* tensor, PD_DataType dtype) {
  PADDLE_ENFORCE_NOT_NULL(tensor,
                          common::errors::InvalidArgument(
                              "The pointer of tensor shouldn't be nullptr"));
  tensor->tensor.dtype = paddle::ConvertToPaddleDType(dtype);
}

void PD_SetPaddleTensorData(PD_Tensor* tensor, PD_PaddleBuf* buf) {
  PADDLE_ENFORCE_NOT_NULL(tensor,
                          common::errors::InvalidArgument(
                              "The pointer of tensor shouldn't be nullptr"));
  tensor->tensor.data = buf->buf;
}

void PD_SetPaddleTensorShape(PD_Tensor* tensor, int* shape, int size) {
  PADDLE_ENFORCE_NOT_NULL(tensor,
                          common::errors::InvalidArgument(
                              "The pointer of tensor shouldn't be nullptr"));
  tensor->tensor.shape.assign(shape, shape + size);
}

const char* PD_GetPaddleTensorName(const PD_Tensor* tensor) {
  PADDLE_ENFORCE_NOT_NULL(tensor,
                          common::errors::InvalidArgument(
                              "The pointer of tensor shouldn't be nullptr"));
  return tensor->tensor.name.c_str();
}

PD_DataType PD_GetPaddleTensorDType(const PD_Tensor* tensor) {
  PADDLE_ENFORCE_NOT_NULL(tensor,
                          common::errors::InvalidArgument(
                              "The pointer of tensor shouldn't be nullptr"));
  return ConvertToPDDataType(tensor->tensor.dtype);
}

PD_PaddleBuf* PD_GetPaddleTensorData(const PD_Tensor* tensor) {
  PADDLE_ENFORCE_NOT_NULL(tensor,
                          common::errors::InvalidArgument(
                              "The pointer of tensor shouldn't be nullptr"));
  PD_PaddleBuf* ret = PD_NewPaddleBuf();
  ret->buf = tensor->tensor.data;
  return ret;
}

const int* PD_GetPaddleTensorShape(const PD_Tensor* tensor, int* size) {
  PADDLE_ENFORCE_NOT_NULL(tensor,
                          common::errors::InvalidArgument(
                              "The pointer of tensor shouldn't be nullptr"));
  const std::vector<int>& shape = tensor->tensor.shape;
  *size = shape.size();
  return shape.data();
}

PD_ZeroCopyTensor* PD_NewZeroCopyTensor() {
  auto* tensor = new PD_ZeroCopyTensor;
  PD_InitZeroCopyTensor(tensor);
  return tensor;
}
void PD_DeleteZeroCopyTensor(PD_ZeroCopyTensor* tensor) {
  if (tensor) {
    PD_DestroyZeroCopyTensor(tensor);
    delete tensor;
  }
  tensor = nullptr;
}

void PD_InitZeroCopyTensor(PD_ZeroCopyTensor* tensor) {
  std::memset(tensor, 0, sizeof(PD_ZeroCopyTensor));
}

void PD_DestroyZeroCopyTensor(PD_ZeroCopyTensor* tensor) {
#define __PADDLE_INFER_CAPI_DELETE_PTR(__ptr) \
  if (__ptr) {                                \
    std::free(__ptr);                         \
    __ptr = nullptr;                          \
  }

  __PADDLE_INFER_CAPI_DELETE_PTR(tensor->data.data);
  __PADDLE_INFER_CAPI_DELETE_PTR(tensor->shape.data);
  __PADDLE_INFER_CAPI_DELETE_PTR(tensor->lod.data);

#undef __PADDLE_INFER_CAPI_DELETE_PTR
}

}  // extern "C"
