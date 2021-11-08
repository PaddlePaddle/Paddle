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

#include "paddle/pten/core/convert_utils.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/platform/gpu_info.h"

namespace pten {

// TODO(chenweihang): Add other place trans cases later
Backend TransToPtenBackend(const paddle::platform::Place& place) {
  if (paddle::platform::is_cpu_place(place)) {
    return Backend::CPU;
  } else if (paddle::platform::is_gpu_place(place)) {
    return Backend::CUDA;
  } else {
    return Backend::UNDEFINED;
  }
}

paddle::experimental::DataType TransToPtenDataType(
    const paddle::framework::proto::VarType::Type& dtype) {
  // Set the order of case branches according to the frequency with
  // the data type is used
  switch (dtype) {
    case paddle::framework::proto::VarType::FP32:
      return DataType::FLOAT32;
    case paddle::framework::proto::VarType::FP64:
      return DataType::FLOAT64;
    case paddle::framework::proto::VarType::INT64:
      return DataType::INT64;
    case paddle::framework::proto::VarType::INT32:
      return DataType::INT32;
    case paddle::framework::proto::VarType::INT8:
      return DataType::INT8;
    case paddle::framework::proto::VarType::UINT8:
      return DataType::UINT8;
    case paddle::framework::proto::VarType::INT16:
      return DataType::INT16;
    case paddle::framework::proto::VarType::COMPLEX64:
      return DataType::COMPLEX64;
    case paddle::framework::proto::VarType::COMPLEX128:
      return DataType::COMPLEX128;
    case paddle::framework::proto::VarType::FP16:
      return DataType::FLOAT16;
    case paddle::framework::proto::VarType::BF16:
      return DataType::BFLOAT16;
    case paddle::framework::proto::VarType::BOOL:
      return DataType::BOOL;
    default:
      return DataType::UNDEFINED;
  }
}

DataLayout TransToPtenDataLayout(const paddle::framework::DataLayout& layout) {
  switch (layout) {
    case paddle::framework::DataLayout::kNHWC:
      return DataLayout::NHWC;
    case paddle::framework::DataLayout::kNCHW:
      return DataLayout::NCHW;
    case paddle::framework::DataLayout::kAnyLayout:
      return DataLayout::ANY;
    case paddle::framework::DataLayout::kMKLDNN:
      return DataLayout::MKLDNN;
    default:
      return DataLayout::UNDEFINED;
  }
}

paddle::platform::Place TransToFluidPlace(const Backend& backend) {
  // TODO(chenweihang): add other trans cases later
  switch (backend) {
    case pten::Backend::CPU:
      return paddle::platform::CPUPlace();
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    case pten::Backend::CUDA:
      return paddle::platform::CUDAPlace(
          paddle::platform::GetCurrentDeviceId());
#endif
#ifdef PADDLE_WITH_MKLDNN
    case pten::Backend::MKLDNN:
      return paddle::platform::CPUPlace();
#endif
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    case pten::Backend::CUDNN:
      return paddle::platform::CUDAPlace(
          paddle::platform::GetCurrentDeviceId());
#endif
    default:
      PADDLE_THROW(paddle::platform::errors::Unimplemented(
          "Unsupported backend `%s` when casting it to paddle place type.",
          backend));
  }
}

paddle::framework::proto::VarType::Type TransToProtoVarType(
    const paddle::experimental::DataType& dtype) {
  // Set the order of case branches according to the frequency with
  // the data type is used
  switch (dtype) {
    case DataType::FLOAT32:
      return paddle::framework::proto::VarType::FP32;
    case DataType::FLOAT64:
      return paddle::framework::proto::VarType::FP64;
    case DataType::INT64:
      return paddle::framework::proto::VarType::INT64;
    case DataType::INT32:
      return paddle::framework::proto::VarType::INT32;
    case DataType::INT8:
      return paddle::framework::proto::VarType::INT8;
    case DataType::UINT8:
      return paddle::framework::proto::VarType::UINT8;
    case DataType::INT16:
      return paddle::framework::proto::VarType::INT16;
    case DataType::COMPLEX64:
      return paddle::framework::proto::VarType::COMPLEX64;
    case DataType::COMPLEX128:
      return paddle::framework::proto::VarType::COMPLEX128;
    case DataType::FLOAT16:
      return paddle::framework::proto::VarType::FP16;
    case DataType::BFLOAT16:
      return paddle::framework::proto::VarType::BF16;
    case DataType::BOOL:
      return paddle::framework::proto::VarType::BOOL;
    default:
      PADDLE_THROW(paddle::platform::errors::Unimplemented(
          "Unsupported data type `%s` when casting it into "
          "paddle data type.",
          dtype));
  }
}

paddle::framework::DataLayout TransToFluidDataLayout(const DataLayout& layout) {
  switch (layout) {
    case DataLayout::NHWC:
      return paddle::framework::DataLayout::kNHWC;
    case DataLayout::NCHW:
      return paddle::framework::DataLayout::kNCHW;
    case DataLayout::ANY:
      return paddle::framework::DataLayout::kAnyLayout;
    case DataLayout::MKLDNN:
      return paddle::framework::DataLayout::kMKLDNN;
    default:
      PADDLE_THROW(paddle::platform::errors::Unimplemented(
          "Unsupported data layout `%s` when casting it into "
          "paddle data layout.",
          layout));
  }
}

}  // namespace pten
