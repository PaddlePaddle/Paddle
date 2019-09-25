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
#include <unordered_map>
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
  tensor->tensor.data = buf;
}

void PD_SetPaddleTensorShape(PD_Tensor* tensor, int* shape, int size) {
  std::vector<int> tensor_shape;
  for (int i = 0; i < size; ++i) {
    tensor_shape.emplace_back(shape[i]);
  }
  tensor->tensor.shape = tensor_shape;
}

void PD_ZeroCopyTensorReshape(PD_ZeroCopyTensor* tensor, int* shape, int size) {
  std::vector<int> new_shape;
  new_shape.assign(shape, shape + size);
  tensor->tensor.Reshape(new_shape);
}

void* PD_ZeroCopyTensorMutableData(PD_ZeroCopyTensor* tensor, PD_Place place) {
  return tensor->tensor.mutable_data<void*>(ConvertToPlace(place));
}

void* PD_ZeroCopyTensorData(PD_ZeroCopyTensor* tensor, PD_Place* place,
                            int* size) {
  return tensor->tensor.data<void*>(ConvertToPlace(place), size);
}

void PD_ZeroCopyTensorCopyToCPU(PD_ZeroCopyTensor* tensor, void* data,
                                PD_DataType data_type) {
  tensor->tensor.copy_to_cpu(
      static_cast<GetDataType<data_type>::RealType*>(data));
}

void PD_ZeroCopyTensorCopyFromCpu(PD_ZeroCopyTensor* tensor, void* data,
                                  PD_DataType data_type) {
  tensor->tensor.copy_from_cpu(
      static_cast<GetDataType<data_type>::RealType*>(data));
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

char* PD_ZeroCopyTensorName(PD_ZeroCopyTensor* tensor) {
  return tensor->tensor.name().data();
}

void PD_SetZeroCopyTensorPlace(PD_ZeroCopyTensor* tensor, PD_Place place,
                               int device = -1) {
  tensor->tensor.SetPlace(ConvertToPlace(place), device);
}

PD_DataType PD_ZeroCopyTensorType(PD_ZeroCopyTensor* tensor) {
  return ConvertToPaddleDType(tensor->tensor.type());
}

}  // extern "C"
