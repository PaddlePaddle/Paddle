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
#include <vector>
#include "paddle/fluid/inference/capi/c_api.h"
#include "paddle/fluid/inference/capi/c_api_internal.h"

extern "C" {

PD_Tensor* PD_NewPaddleTensor() { return new PD_Tensor; }

void PD_DeletePaddleTensor(PD_Tensor* tensor) {
  if (tensor) {
    delete tensor;
    tensor = nullptr;
  }
}

void PD_SetPaddleTensorName(PD_Tensor* tensor, char* name) {
  tensor->tensor.name = std::string(name);
}

void PD_SetPaddleTensorDType(PD_Tensor* tensor, PD_DataType dtype) {
  tensor->tensor.dtype = ConvertToPaddleDType(dtype);
}

void PD_SetPaddleTensorData(PD_Tensor* tensor, PD_PaddleBuf* buf) {
  tensor->tensor.data = buf->buf;
}

void PD_SetPaddleTensorShape(PD_Tensor* tensor, int* shape, int size) {
  tensor->tensor.shape.assign(shape, shape + size);
}

void PD_ZeroCopyTensorReshape(PD_ZeroCopyTensor* tensor, int* shape, int size) {
  std::vector<int> new_shape;
  new_shape.assign(shape, shape + size);
  tensor->tensor.Reshape(new_shape);
}

void* PD_ZeroCopyTensorMutableData(PD_ZeroCopyTensor* tensor, PD_Place place) {
  return tensor->tensor.mutable_data<void*>(ConvertToPlace(place));
}

void* PD_ZeroCopyTensorData(PD_ZeroCopyTensor* tensor, PD_Place place,
                            int* size) {
  paddle::PaddlePlace p = ConvertToPlace(place);
  return tensor->tensor.data<void*>(&p, size);
}

void PD_ZeroCopyTensorCopyToCPU(PD_ZeroCopyTensor* tensor, void* data,
                                PD_DataType data_type) {
  switch (data_type) {
    case PD_FLOAT32:
      tensor->tensor.copy_to_cpu(static_cast<float*>(data));
      break;
    case PD_INT32:
      tensor->tensor.copy_to_cpu(static_cast<int32_t*>(data));
      break;
    case PD_INT64:
      tensor->tensor.copy_to_cpu(static_cast<int64_t*>(data));
      break;
    case PD_UINT8:
      tensor->tensor.copy_to_cpu(static_cast<uint8_t*>(data));
      break;
    default:
      PADDLE_ENFORCE(false, "Unsupport dtype.");
      break;
  }
}

void PD_ZeroCopyTensorCopyFromCpu(PD_ZeroCopyTensor* tensor, void* data,
                                  PD_DataType data_type) {
  switch (data_type) {
    case PD_FLOAT32:
      tensor->tensor.copy_from_cpu(static_cast<float*>(data));
      break;
    case PD_INT32:
      tensor->tensor.copy_from_cpu(static_cast<int32_t*>(data));
      break;
    case PD_INT64:
      tensor->tensor.copy_from_cpu(static_cast<int64_t*>(data));
      break;
    case PD_UINT8:
      tensor->tensor.copy_from_cpu(static_cast<uint8_t*>(data));
      break;
    default:
      PADDLE_ENFORCE(false, "Unsupport dtype.");
      break;
  }
}

int* PD_ZeroCopyTensorShape(PD_ZeroCopyTensor* tensor, int* size) {
  std::vector<int> ret_shape;
  ret_shape = tensor->tensor.shape();
  int s = ret_shape.size();
  int* shapes;
  for (int i = 0; i < s; ++i) {
    shapes[i] = ret_shape[i];
  }
  size = &s;
  return shapes;
}

const char* PD_ZeroCopyTensorName(PD_ZeroCopyTensor* tensor) {
  return tensor->tensor.name().c_str();
}

void PD_SetZeroCopyTensorPlace(PD_ZeroCopyTensor* tensor, PD_Place place,
                               int device) {
  tensor->tensor.SetPlace(ConvertToPlace(place), device);
}

PD_DataType PD_ZeroCopyTensorType(PD_ZeroCopyTensor* tensor) {
  return ConvertToPDDataType(tensor->tensor.type());
}

}  // extern "C"
