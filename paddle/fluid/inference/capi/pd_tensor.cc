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

using paddle::ConvertToPaddleDType;
using paddle::ConvertToPlace;
using paddle::ConvertToPDDataType;
using paddle::ConvertToACPrecision;

namespace paddle {
paddle::PaddleDType ConvertToPaddleDType(PD_DataType dtype) {
  switch (dtype) {
    case PD_FLOAT32:
      return PD_PaddleDType::FLOAT32;
    case PD_INT32:
      return PD_PaddleDType::INT32;
    case PD_INT64:
      return PD_PaddleDType::INT64;
    case PD_UINT8:
      return PD_PaddleDType::UINT8;
    default:
      PADDLE_ENFORCE(false, "Unsupport dtype.");
      return PD_PaddleDType::FLOAT32;
  }
  PADDLE_ENFORCE(false, "Unsupport dtype.");
  return PD_PaddleDType::FLOAT32;
}

paddle::PaddlePlace ConvertToPlace(PD_Place dtype) {
  switch (dtype) {
    case PD_UNK:
      return PD_PaddlePlace::kUNK;
    case PD_CPU:
      return PD_PaddlePlace::kCPU;
    case PD_GPU:
      return PD_PaddlePlace::kGPU;
    default:
      PADDLE_ENFORCE(false, "Unsupport place.");
      return PD_PaddlePlace::kUNK;
  }
  PADDLE_ENFORCE(false, "Unsupport dtype.");
  return PD_PaddlePlace::kUNK;
}

PD_DataType ConvertToPDDataType(PD_PaddleDType dtype) {
  switch (dtype) {
    case PD_PaddleDType::FLOAT32:
      return PD_DataType::PD_FLOAT32;
    case PD_PaddleDType::INT32:
      return PD_DataType::PD_INT32;
    case PD_PaddleDType::INT64:
      return PD_DataType::PD_INT64;
    case PD_PaddleDType::UINT8:
      return PD_DataType::PD_UINT8;
    default:
      PADDLE_ENFORCE(false, "Unsupport place.");
      return PD_DataType::PD_UNKDTYPE;
  }
  PADDLE_ENFORCE(false, "Unsupport place.");
  return PD_DataType::PD_UNKDTYPE;
}

PD_ACPrecision ConvertToACPrecision(Precision dtype) {
  switch (dtype) {
    case Precision::kFloat32:
      return PD_ACPrecision::kFloat32;
    case Precision::kInt8:
      return PD_ACPrecision::kInt8;
    case Precision::kHalf:
      return PD_ACPrecision::kHalf;
    default:
      PADDLE_ENFORCE(false, "Unsupport place.");
      return PD_ACPrecision::kFloat32;
  }
  PADDLE_ENFORCE(false, "Unsupport place.");
  return PD_ACPrecision::kFloat32;
}
}  // namespace paddle

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
  tensor->tensor.dtype = paddle::ConvertToPaddleDType(dtype);
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

float* PD_ZeroCopyTensorMutableFLOATData(PD_ZeroCopyTensor* tensor,
                                         PD_Place place) {
  return tensor->tensor.mutable_data<float>(paddle::ConvertToPlace(place));
}
int32_t* PD_ZeroCopyTensorMutableINT32Data(PD_ZeroCopyTensor* tensor,
                                           PD_Place place) {
  return tensor->tensor.mutable_data<int32_t>(paddle::ConvertToPlace(place));
}
int64_t* PD_ZeroCopyTensorMutableINT64Data(PD_ZeroCopyTensor* tensor,
                                           PD_Place place) {
  return tensor->tensor.mutable_data<int64_t>(paddle::ConvertToPlace(place));
}
uint8_t* PD_ZeroCopyTensorMutableUINT8Data(PD_ZeroCopyTensor* tensor,
                                           PD_Place place) {
  return tensor->tensor.mutable_data<uint8_t>(paddle::ConvertToPlace(place));
}

float* PD_ZeroCopyTensorFLOATData(PD_ZeroCopyTensor* tensor, PD_Place place,
                                  int* size) {
  paddle::PaddlePlace p = paddle::ConvertToPlace(place);
  return tensor->tensor.data<float>(&p, size);
}
int32_t* PD_ZeroCopyTensorINT32Data(PD_ZeroCopyTensor* tensor, PD_Place place,
                                    int* size) {
  paddle::PaddlePlace p = paddle::ConvertToPlace(place);
  return tensor->tensor.data<int32_t>(&p, size);
}
int64_t* PD_ZeroCopyTensorINT64Data(PD_ZeroCopyTensor* tensor, PD_Place place,
                                    int* size) {
  paddle::PaddlePlace p = paddle::ConvertToPlace(place);
  return tensor->tensor.data<int64_t>(&p, size);
}
uint8_t* PD_ZeroCopyTensorUINT8Data(PD_ZeroCopyTensor* tensor, PD_Place place,
                                    int* size) {
  paddle::PaddlePlace p = paddle::ConvertToPlace(place);
  return tensor->tensor.data<uint8_t>(&p, size);
}

void PD_ZeroCopyToCPU(PD_ZeroCopyTensor* tensor, void* data,
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

void PD_ZeroCopyFromCpu(PD_ZeroCopyTensor* tensor, void* data,
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

int* PD_ZeroCopyTensorShape(PD_ZeroCopyTensor* tensor, int** size) {
  std::vector<int> ret_shape;
  ret_shape = tensor->tensor.shape();
  int s = ret_shape.size();
  int* shapes = new int[s];
  for (int i = 0; i < s; ++i) {
    shapes[i] = ret_shape[i];
  }
  *size = &s;
  return shapes;
}

const char* PD_ZeroCopyTensorName(PD_ZeroCopyTensor* tensor) {
  return tensor->tensor.name().c_str();
}

void PD_SetZeroCopyTensorPlace(PD_ZeroCopyTensor* tensor, PD_Place place,
                               int device) {
  tensor->tensor.SetPlace(paddle::ConvertToPlace(place), device);
}

PD_DataType PD_ZeroCopyTensorType(PD_ZeroCopyTensor* tensor) {
  return paddle::ConvertToPDDataType(tensor->tensor.type());
}

}  // extern "C"
