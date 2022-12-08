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

#include "paddle/fluid/platform/device/ipu/ipu_utils.h"

#include <cmath>

namespace paddle {
namespace platform {
namespace ipu {

const popart::DataType VarType2PopartDType(const VarType::Type type) {
  switch (type) {
    case VarType::UINT8:
      return popart::DataType::UINT8;
    case VarType::INT8:
      return popart::DataType::INT8;
    case VarType::INT16:
      return popart::DataType::INT16;
    case VarType::INT32:
      return popart::DataType::INT32;
    case VarType::INT64:
      return popart::DataType::INT64;
    case VarType::BOOL:
      return popart::DataType::BOOL;
    case VarType::FP64:
      return popart::DataType::DOUBLE;
    case VarType::FP32:
      return popart::DataType::FLOAT;
    case VarType::FP16:
      return popart::DataType::FLOAT16;
    case VarType::BF16:
      return popart::DataType::BFLOAT16;
    case VarType::COMPLEX64:
      return popart::DataType::COMPLEX64;
    case VarType::COMPLEX128:
      return popart::DataType::COMPLEX128;
    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "Unsupported VarType::Type when converting to popart data type."));
  }
}

const popart::DataType PhiDType2PopartDType(const phi::DataType type) {
  switch (type) {
    case phi::DataType::UINT8:
      return popart::DataType::UINT8;
    case phi::DataType::INT8:
      return popart::DataType::INT8;
    case phi::DataType::INT16:
      return popart::DataType::INT16;
    case phi::DataType::INT32:
      return popart::DataType::INT32;
    case phi::DataType::INT64:
      return popart::DataType::INT64;
    case phi::DataType::BOOL:
      return popart::DataType::BOOL;
    case phi::DataType::FLOAT64:
      return popart::DataType::DOUBLE;
    case phi::DataType::FLOAT32:
      return popart::DataType::FLOAT;
    case phi::DataType::FLOAT16:
      return popart::DataType::FLOAT16;
    case phi::DataType::BFLOAT16:
      return popart::DataType::BFLOAT16;
    case phi::DataType::COMPLEX64:
      return popart::DataType::COMPLEX64;
    case phi::DataType::COMPLEX128:
      return popart::DataType::COMPLEX128;
    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "Unsupported phi::DataType when converting to popart data type."));
  }
}

const phi::DataType PopefDtype2PhiDtype(const popef::DataType type) {
  switch (type) {
    case popef::DataType::U8:
      return phi::DataType::UINT8;
    case popef::DataType::S8:
      return phi::DataType::INT8;
    case popef::DataType::S16:
      return phi::DataType::INT16;
    case popef::DataType::S32:
      return phi::DataType::INT32;
    case popef::DataType::S64:
      return phi::DataType::INT64;
    case popef::DataType::BOOL:
      return phi::DataType::BOOL;
    case popef::DataType::F64:
      return phi::DataType::FLOAT64;
    case popef::DataType::F32:
      return phi::DataType::FLOAT32;
    case popef::DataType::F16:
      return phi::DataType::FLOAT16;
    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "Unsupported phi::DataType when converting to popef data type."));
  }
}

const VarType::Type PopartDType2VarType(const popart::DataType type) {
  switch (type) {
    case popart::DataType::UINT8:
      return VarType::UINT8;
    case popart::DataType::INT8:
      return VarType::INT8;
    case popart::DataType::INT16:
      return VarType::INT16;
    case popart::DataType::INT32:
      return VarType::INT32;
    case popart::DataType::INT64:
      return VarType::INT64;
    case popart::DataType::BOOL:
      return VarType::BOOL;
    case popart::DataType::DOUBLE:
      return VarType::FP64;
    case popart::DataType::FLOAT:
      return VarType::FP32;
    case popart::DataType::FLOAT16:
      return VarType::FP16;
    case popart::DataType::BFLOAT16:
      return VarType::BF16;
    case popart::DataType::COMPLEX64:
      return VarType::COMPLEX64;
    case popart::DataType::COMPLEX128:
      return VarType::COMPLEX128;
    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "Unsupported popart::DataType when converting to var type."));
  }
}

const VarType::Type PopefDType2VarType(const popef::DataType type) {
  switch (type) {
    case popef::DataType::U8:
      return VarType::UINT8;
    case popef::DataType::S8:
      return VarType::INT8;
    case popef::DataType::S16:
      return VarType::INT16;
    case popef::DataType::S32:
      return VarType::INT32;
    case popef::DataType::S64:
      return VarType::INT64;
    case popef::DataType::BOOL:
      return VarType::BOOL;
    case popef::DataType::F64:
      return VarType::FP64;
    case popef::DataType::F32:
      return VarType::FP32;
    case popef::DataType::F16:
      return VarType::FP16;
    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "Unsupported popart::DataType when converting to var type."));
  }
}

const popart::DataType OnnxDType2PopartType(const ONNXDataType type) {
  switch (type) {
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
          "Unsupported ONNXDataType when converting to popart data type."));
  }
}

const ONNXDataType VarType2OnnxDType(const VarType::Type type) {
  switch (type) {
    case VarType::BOOL:
      return ONNXDataType::BOOL;
    case VarType::INT16:
      return ONNXDataType::INT16;
    case VarType::INT32:
      return ONNXDataType::INT32;
    case VarType::INT64:
      return ONNXDataType::INT64;
    case VarType::FP16:
      return ONNXDataType::FLOAT16;
    case VarType::FP32:
      return ONNXDataType::FLOAT;
    case VarType::FP64:
      return ONNXDataType::DOUBLE;
    case VarType::UINT8:
      return ONNXDataType::UINT8;
    case VarType::INT8:
      return ONNXDataType::INT8;
    case VarType::BF16:
      return ONNXDataType::BFLOAT16;
    case VarType::COMPLEX64:
      return ONNXDataType::COMPLEX64;
    case VarType::COMPLEX128:
      return ONNXDataType::COMPLEX128;
    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "Unsupported VarType::Type when converting to onnx data type."));
  }
}

const std::string VarType2PopartStr(const VarType::Type type) {
  switch (type) {
    case VarType::UINT8:
      return "UINT8";
    case VarType::INT8:
      return "INT8";
    case VarType::INT16:
      return "INT16";
    case VarType::INT32:
      return "INT32";
    case VarType::INT64:
      return "INT64";
    case VarType::BOOL:
      return "BOOL";
    case VarType::FP64:
      return "DOUBLE";
    case VarType::FP32:
      return "FLOAT";
    case VarType::FP16:
      return "FLOAT16";
    default:
      PADDLE_THROW(platform::errors::Unavailable(
          "Unsupported VarType::Type when converting to popart type string."));
  }
}

const bool GetBoolEnv(const std::string& str) {
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

const int RequestIpus(const int num_ipus) {
  return std::pow(2, ceil(log2(num_ipus)));
}

std::shared_ptr<popef::Model> PopartSessionToPopefModel(
    popart::Session* session) {
  VLOG(10) << "Converting popart session to popef model";
  auto temp_stream = std::make_shared<std::stringstream>();
  session->compileAndExport(*temp_stream);
  auto reader = std::make_shared<popef::Reader>();
  reader->parseStream(temp_stream);
  return popef::ModelBuilder(reader).createModel();
}

}  // namespace ipu
}  // namespace platform
}  // namespace paddle
