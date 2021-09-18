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

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarraytypes.h>

#include "paddle/fluid/platform/complex.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/top/core/convert_utils.h"

namespace pt {

paddle::platform::Place TransToFluidPlace(const Backend& backend) {
  // TODO(chenweihang): add other trans cases
  switch (backend) {
    case pt::Backend::kCPU:
      return paddle::platform::CPUPlace();
    case pt::Backend::kCUDA:
      return paddle::platform::CUDAPlace();
    case pt::Backend::kXPU:
      return paddle::platform::XPUPlace();
    case pt::Backend::kNPU:
      return paddle::platform::NPUPlace();
    case pt::Backend::kMKLDNN:
      return paddle::platform::CPUPlace();
    case pt::Backend::kCUDNN:
      return paddle::platform::CUDAPlace();
    default:
      PADDLE_THROW(paddle::platform::errors::Unimplemented(
          "Unsupported backend when casting it to paddle place type."));
  }
}

// TODO(chenweihang): Add other place branchs
Backend TransToPtenBackend(const paddle::platform::Place& place) {
  if (paddle::platform::is_cpu_place(place)) {
    return Backend::kCPU;
  } else if (paddle::platform::is_gpu_place(place)) {
    return Backend::kCUDA;
  } else if (paddle::platform::is_cuda_pinned_place(place)) {
    return Backend::kCUDAPinned;
  } else if (paddle::platform::is_xpu_place(place)) {
    return Backend::kXPU;
  } else if (paddle::platform::is_npu_place(place)) {
    return Backend::kNPU;
  } else if (paddle::platform::is_npu_pinned_place(place)) {
    return Backend::kNPUPinned;
  } else {
    return Backend::kUndef;
  }
}

pt::DataType TransToPtenDataType(
    const paddle::framework::proto::VarType::Type& dtype) {
  // Set the order of case branches according to the frequency with
  // the data type is used
  switch (dtype) {
    case paddle::framework::proto::VarType::FP32:
      return DataType::kFLOAT32;
    case paddle::framework::proto::VarType::FP64:
      return DataType::kFLOAT64;
    case paddle::framework::proto::VarType::INT64:
      return DataType::kINT64;
    case paddle::framework::proto::VarType::INT32:
      return DataType::kINT32;
    case paddle::framework::proto::VarType::INT8:
      return DataType::kINT8;
    case paddle::framework::proto::VarType::UINT8:
      return DataType::kUINT8;
    case paddle::framework::proto::VarType::INT16:
      return DataType::kINT16;
    case paddle::framework::proto::VarType::COMPLEX64:
      return DataType::kCOMPLEX64;
    case paddle::framework::proto::VarType::COMPLEX128:
      return DataType::kCOMPLEX128;
    case paddle::framework::proto::VarType::FP16:
      return DataType::kFLOAT16;
    case paddle::framework::proto::VarType::BOOL:
      return DataType::kBOOL;
    default:
      return DataType::kUndef;
  }
}

DataLayout TransToPtenLayout(const paddle::framework::DataLayout& layout) {
  switch (layout) {
    case paddle::framework::DataLayout::kNHWC:
      return DataLayout::kNHWC;
    case paddle::framework::DataLayout::kNCHW:
      return DataLayout::kNCHW;
    case paddle::framework::DataLayout::kAnyLayout:
      return DataLayout::kAny;
    case paddle::framework::DataLayout::kMKLDNN:
      return DataLayout::kMKLDNN;
    default:
      return DataLayout::kUndef;
  }
}

paddle::framework::proto::VarType::Type TransToProtoVarType(
    const pt::DataType& dtype) {
  // Set the order of case branches according to the frequency with
  // the data type is used
  switch (dtype) {
    case DataType::kFLOAT32:
      return paddle::framework::proto::VarType::FP32;
    case DataType::kFLOAT64:
      return paddle::framework::proto::VarType::FP64;
    case DataType::kINT64:
      return paddle::framework::proto::VarType::INT64;
    case DataType::kINT32:
      return paddle::framework::proto::VarType::INT32;
    case DataType::kINT8:
      return paddle::framework::proto::VarType::INT8;
    case DataType::kUINT8:
      return paddle::framework::proto::VarType::UINT8;
    case DataType::kINT16:
      return paddle::framework::proto::VarType::INT16;
    case DataType::kCOMPLEX64:
      return paddle::framework::proto::VarType::COMPLEX64;
    case DataType::kCOMPLEX128:
      return paddle::framework::proto::VarType::COMPLEX128;
    case DataType::kFLOAT16:
      return paddle::framework::proto::VarType::FP16;
    case DataType::kBOOL:
      return paddle::framework::proto::VarType::BOOL;
    default:
      PADDLE_THROW(paddle::platform::errors::Unimplemented(
          "Unsupported data type code(%d) when casting enum data type into "
          "paddle data type.",
          static_cast<int>(dtype)));
  }
}

size_t DataTypeSize(DataType dtype) {
  switch (dtype) {
    case DataType::kUndef:
      return 0;
    case DataType::kBOOL:
      return sizeof(bool);
    case DataType::kINT8:
      return sizeof(int8_t);
    case DataType::kUINT8:
      return sizeof(uint8_t);
    case DataType::kINT16:
      return sizeof(int16_t);
    case DataType::kUINT16:
      return sizeof(uint16_t);
    case DataType::kINT32:
      return sizeof(int);
    case DataType::kINT64:
      return sizeof(int64_t);
    case DataType::kFLOAT16:
      return sizeof(paddle::platform::float16);
    case DataType::kFLOAT32:
      return sizeof(float);
    case DataType::kFLOAT64:
      return sizeof(double);
    case DataType::kCOMPLEX64:
      return sizeof(paddle::platform::complex<float>);
    case DataType::kCOMPLEX128:
      return sizeof(paddle::platform::complex<double>);
    default:
      return 0;
  }
}

DataType String2DataTyep(const std::string& str) {
  if (str == "bool") {
    return DataType::kBOOL;
  } else if (str == "float16") {
    return DataType::kFLOAT16;
  } else if (str == "uint16") {
    return DataType::kUINT16;
  } else if (str == "float32") {
    return DataType::kFLOAT32;
  } else if (str == "float64") {
    return DataType::kFLOAT64;
  } else if (str == "int8") {
    return DataType::kINT8;
  } else if (str == "int16") {
    return DataType::kINT16;
  } else if (str == "int32") {
    return DataType::kINT32;
  } else if (str == "int64") {
    return DataType::kINT64;
  } else if (str == "uint8") {
    return DataType::kUINT8;
  } else if (str == "complex64") {
    return DataType::kCOMPLEX64;
  } else if (str == "complex128") {
    return DataType::kCOMPLEX128;
  } else {
    return DataType::kUndef;
  }
}

std::string DataType2String(DataType dtype) {
  switch (dtype) {
    case DataType::kBOOL:
      return "bool";
    case DataType::kINT8:
      return "int8";
    case DataType::kUINT8:
      return "uint8";
    case DataType::kINT16:
      return "int16";
    case DataType::kUINT16:
      return "uint16";
    case DataType::kINT32:
      return "int32";
    case DataType::kINT64:
      return "int64";
    case DataType::kFLOAT16:
      return "float16";
    case DataType::kFLOAT32:
      return "float32";
    case DataType::kFLOAT64:
      return "float64";
    case DataType::kCOMPLEX64:
      return "complex64";
    case DataType::kCOMPLEX128:
      return "complex128";
    default:
      PADDLE_THROW(paddle::platform::errors::InvalidArgument(
          "Unknow pt::DataType, the int value = %d.", static_cast<int>(dtype)));
      return "";
  }
}

int TensorDtype2NumpyDtype(pt::DataType dtype) {
  switch (dtype) {
    case pt::DataType::kBOOL:
      return NPY_TYPES::NPY_BOOL;
    case pt::DataType::kINT8:
      return NPY_TYPES::NPY_INT8;
    case pt::DataType::kUINT8:
      return NPY_TYPES::NPY_UINT8;
    case pt::DataType::kINT16:
      return NPY_TYPES::NPY_INT16;
    case pt::DataType::kUINT16:
      return NPY_TYPES::NPY_UINT16;
    case pt::DataType::kINT32:
      return NPY_TYPES::NPY_INT32;
    case pt::DataType::kINT64:
      return NPY_TYPES::NPY_INT64;
    case pt::DataType::kFLOAT16:
      return NPY_TYPES::NPY_FLOAT;  // numpy not have float16
    case pt::DataType::kFLOAT32:
      return NPY_TYPES::NPY_FLOAT;
    case pt::DataType::kFLOAT64:
      return NPY_TYPES::NPY_DOUBLE;
    case pt::DataType::kCOMPLEX64:
      return NPY_TYPES::NPY_COMPLEX64;
    case pt::DataType::kCOMPLEX128:
      return NPY_TYPES::NPY_COMPLEX128;
    default:
      PADDLE_THROW(paddle::platform::errors::InvalidArgument(
          "Unknow pt::DataType, the int value = %d.", static_cast<int>(dtype)));
      return 0;
  }
}

}  // namespace pt
