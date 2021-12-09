/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/ipu/ipu_utils.h"

namespace paddle {
namespace platform {
namespace ipu {

void* PaddleIArray::data() { return tensor_->data<void>(); }

popart::DataType PaddleIArray::dataType() const {
  return VarType2PopartType(tensor_->type());
}

std::size_t PaddleIArray::rank() const { return tensor_->dims().size(); }

int64_t PaddleIArray::dim(size_t index) const {
  return tensor_->dims().at(index);
}

std::size_t PaddleIArray::nelms() const {
  return std::accumulate(shape_.begin(), shape_.end(), static_cast<int64_t>(1),
                         std::multiplies<int64_t>());
}

const popart::Shape PaddleIArray::shape() const { return shape_; }

popart::DataType VarType2PopartType(
    const framework::proto::VarType::Type type) {
  switch (type) {
    case framework::proto::VarType::UINT8:
      return popart::DataType::UINT8;
    case framework::proto::VarType::INT8:
      return popart::DataType::INT8;
    case framework::proto::VarType::INT16:
      return popart::DataType::INT16;
    case framework::proto::VarType::INT32:
      return popart::DataType::INT32;
    case framework::proto::VarType::INT64:
      return popart::DataType::INT64;
    case framework::proto::VarType::BOOL:
      return popart::DataType::BOOL;
    case framework::proto::VarType::FP64:
      return popart::DataType::DOUBLE;
    case framework::proto::VarType::FP32:
      return popart::DataType::FLOAT;
    case framework::proto::VarType::FP16:
      return popart::DataType::FLOAT16;
    case framework::proto::VarType::BF16:
      return popart::DataType::BFLOAT16;
    case framework::proto::VarType::COMPLEX64:
      return popart::DataType::COMPLEX64;
    case framework::proto::VarType::COMPLEX128:
      return popart::DataType::COMPLEX128;
    default:
      PADDLE_THROW(paddle::platform::errors::Unavailable(
          "Unsupported Paddle var type."));
  }
}

framework::proto::VarType::Type PopartType2VarType(
    const popart::DataType type) {
  switch (type) {
    case popart::DataType::UINT8:
      return framework::proto::VarType::UINT8;
    case popart::DataType::INT8:
      return framework::proto::VarType::INT8;
    case popart::DataType::INT16:
      return framework::proto::VarType::INT16;
    case popart::DataType::INT32:
      return framework::proto::VarType::INT32;
    case popart::DataType::INT64:
      return framework::proto::VarType::INT64;
    case popart::DataType::BOOL:
      return framework::proto::VarType::BOOL;
    case popart::DataType::DOUBLE:
      return framework::proto::VarType::FP64;
    case popart::DataType::FLOAT:
      return framework::proto::VarType::FP32;
    case popart::DataType::FLOAT16:
      return framework::proto::VarType::FP16;
    case popart::DataType::BFLOAT16:
      return framework::proto::VarType::BF16;
    case popart::DataType::COMPLEX64:
      return framework::proto::VarType::COMPLEX64;
    case popart::DataType::COMPLEX128:
      return framework::proto::VarType::COMPLEX128;
    default:
      PADDLE_THROW(paddle::platform::errors::Unavailable(
          "Unsupported Paddle var type."));
  }
}

popart::DataType OnnxDtype2PopartType(const int type) {
  auto dtype = static_cast<ONNXDataType>(type);
  switch (dtype) {
    case ONNXDataType::BOOL:
      return popart::DataType::BOOL;
    case ONNXDataType::INT16:
      return popart::DataType::INT16;
    case ONNXDataType::INT32:
      return popart::DataType::INT32;
    case ONNXDataType::INT64:
      return popart::DataType::INT64;
    case ONNXDataType::FLOAT16:
      return popart::DataType::FLOAT16;
    case ONNXDataType::FLOAT:
      return popart::DataType::FLOAT;
    case ONNXDataType::DOUBLE:
      return popart::DataType::DOUBLE;
    case ONNXDataType::UINT8:
      return popart::DataType::UINT8;
    case ONNXDataType::INT8:
      return popart::DataType::INT8;
    case ONNXDataType::BFLOAT16:
      return popart::DataType::BFLOAT16;
    case ONNXDataType::COMPLEX64:
      return popart::DataType::COMPLEX64;
    case ONNXDataType::COMPLEX128:
      return popart::DataType::COMPLEX128;
    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "Unsupported ONNX data type: %d.", dtype));
  }
}

// count num should > 0
bool GetBoolEnv(std::string str) {
  char* str_val = getenv(str.c_str());
  if (str_val == NULL) {
    return false;
  } else {
    bool val = false;
    if (strcmp(str_val, "1") == 0 || strcmp(str_val, "true") == 0 ||
        strcmp(str_val, "True") == 0 || strcmp(str_val, "TRUE") == 0)
      val = true;
    return val;
  }
}

}  // namespace ipu
}  // namespace platform
}  // namespace paddle
