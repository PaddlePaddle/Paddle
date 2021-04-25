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

#pragma once

#include <memory>

#include "paddle/fluid/extension/include/ext_tensor.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/platform/gpu_info.h"
#include "paddle/fluid/platform/place.h"
#ifdef PADDLE_WITH_CUDA
#endif
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace framework {

class CustomTensorUtils {
 public:
  /// \brief Share data TO another tensor.
  /// Use this to pass tensor from op to op
  /// \return void.
  static void ShareDataTo(const paddle::Tensor& src, void* dst);

  /// \brief Share data FROM another tensor.
  /// Use this to pass tensor from op to op
  /// \return void.
  static void ShareDataFrom(const void* src, const paddle::Tensor& dst);

  static framework::proto::VarType::Type ConvertEnumDTypeToInnerDType(
      const paddle::DataType& dtype) {
    switch (dtype) {
      case paddle::DataType::FLOAT64:
        return framework::proto::VarType::FP64;
      case paddle::DataType::FLOAT32:
        return framework::proto::VarType::FP32;
      case paddle::DataType::UINT8:
        return framework::proto::VarType::UINT8;
      case paddle::DataType::INT8:
        return framework::proto::VarType::INT8;
      case paddle::DataType::INT32:
        return framework::proto::VarType::INT32;
      case paddle::DataType::INT64:
        return framework::proto::VarType::INT64;
      case paddle::DataType::INT16:
        return framework::proto::VarType::INT16;
      case paddle::DataType::COMPLEX64:
        return framework::proto::VarType::COMPLEX64;
      case paddle::DataType::COMPLEX128:
        return framework::proto::VarType::COMPLEX128;
      case paddle::DataType::FLOAT16:
        return framework::proto::VarType::FP16;
      case paddle::DataType::BOOL:
        return framework::proto::VarType::BOOL;
      default:
        PADDLE_THROW(platform::errors::Unimplemented(
            "Unsupported data type code(%d) when casting enum data type into "
            "paddle data type.",
            static_cast<int>(dtype)));
    }
  }

  static paddle::DataType ConvertInnerDTypeToEnumDType(
      const framework::proto::VarType::Type& dtype) {
    switch (dtype) {
      case framework::proto::VarType::FP64:
        return paddle::DataType::FLOAT64;
      case framework::proto::VarType::FP32:
        return paddle::DataType::FLOAT32;
      case framework::proto::VarType::INT64:
        return paddle::DataType::INT64;
      case framework::proto::VarType::INT32:
        return paddle::DataType::INT32;
      case framework::proto::VarType::INT8:
        return paddle::DataType::INT8;
      case framework::proto::VarType::UINT8:
        return paddle::DataType::UINT8;
      case framework::proto::VarType::INT16:
        return paddle::DataType::INT16;
      case framework::proto::VarType::COMPLEX64:
        return paddle::DataType::COMPLEX64;
      case framework::proto::VarType::COMPLEX128:
        return paddle::DataType::COMPLEX128;
      case framework::proto::VarType::FP16:
        return paddle::DataType::FLOAT16;
      case framework::proto::VarType::BOOL:
        return paddle::DataType::BOOL;
      default:
        PADDLE_THROW(platform::errors::Unimplemented(
            "Unsupported data type `%s` when casting paddle data type into "
            "enum data type.",
            DataTypeToString(dtype)));
    }
  }

  // PaddlePlace <-> platform::Place
  static platform::Place ConvertEnumPlaceToInnerPlace(const PlaceType& pc) {
    if (pc == PlaceType::kCPU) {
      return platform::Place(platform::CPUPlace());
    } else if (pc == PlaceType::kGPU) {
#ifdef PADDLE_WITH_CUDA
      return platform::Place(
          platform::CUDAPlace(platform::GetCurrentDeviceId()));
#endif
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Unsupported place type code(%d) when "
          "casting enum place to paddle place.",
          static_cast<int>(pc)));
    }
    return platform::Place();
  }

  static PlaceType ConvertInnerPlaceToEnumPlace(const platform::Place& pc) {
    if (platform::is_cpu_place(pc)) {
      return PlaceType::kCPU;
    } else if (platform::is_gpu_place(pc)) {
#ifdef PADDLE_WITH_CUDA
      return PlaceType::kGPU;
#endif
    } else {
      PADDLE_THROW(
          platform::errors::Unimplemented("Unsupported place type `%s` when "
                                          "casting paddle place to enum place.",
                                          pc));
    }
    return PlaceType::kUNK;
  }

  static void SetTensorCurrentStream(paddle::Tensor* src,
                                     const platform::Place& pc) {
    if (platform::is_gpu_place(pc)) {
#ifdef PADDLE_WITH_CUDA
      auto* dev_ctx = static_cast<platform::CUDADeviceContext*>(
          platform::DeviceContextPool::Instance().Get(pc));
      src->stream_.SetStream(reinterpret_cast<void*>(dev_ctx->stream()));
#endif
    } else {
      return;
    }
  }
};

}  // namespace framework
}  // namespace paddle
