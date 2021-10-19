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

#include "paddle/tcmpt/core/convert_utils.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/platform/gpu_info.h"

namespace pt {

// TODO(chenweihang): Add other place branchs
Backend TransToPtBackend(const paddle::platform::Place& place) {
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

pt::DataType TransToPtDataType(
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
    case paddle::framework::proto::VarType::BF16:
      return DataType::kBFLOAT16;
    case paddle::framework::proto::VarType::BOOL:
      return DataType::kBOOL;
    default:
      return DataType::kUndef;
  }
}

DataLayout TransToPtDataLayout(const paddle::framework::DataLayout& layout) {
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

paddle::platform::Place TransToFluidPlace(const Backend& backend) {
  // TODO(chenweihang): add other trans cases
  switch (backend) {
    case pt::Backend::kCPU:
      return paddle::platform::CPUPlace();
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    case pt::Backend::kCUDA:
      return paddle::platform::CUDAPlace(
          paddle::platform::GetCurrentDeviceId());
#endif
#ifdef PADDLE_WITH_XPU
    case pt::Backend::kXPU:
      // TODO(chenweihang): add device id
      return paddle::platform::XPUPlace();
#endif
#ifdef PADDLE_WITH_NPU
    case pt::Backend::kNPU:
      // TODO(chenweihang): add device id
      return paddle::platform::NPUPlace();
#endif
#ifdef PADDLE_WITH_MKLDNN
    case pt::Backend::kMKLDNN:
      return paddle::platform::CPUPlace();
#endif
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    case pt::Backend::kCUDNN:
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
    case DataType::kBFLOAT16:
      return paddle::framework::proto::VarType::BF16;
    case DataType::kBOOL:
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
    case DataLayout::kNHWC:
      return paddle::framework::DataLayout::kNHWC;
    case DataLayout::kNCHW:
      return paddle::framework::DataLayout::kNCHW;
    case DataLayout::kAny:
      return paddle::framework::DataLayout::kAnyLayout;
    case DataLayout::kMKLDNN:
      return paddle::framework::DataLayout::kMKLDNN;
    default:
      PADDLE_THROW(paddle::platform::errors::Unimplemented(
          "Unsupported data layout `%s` when casting it into "
          "paddle data layout.",
          layout));
  }
}

}  // namespace pt
