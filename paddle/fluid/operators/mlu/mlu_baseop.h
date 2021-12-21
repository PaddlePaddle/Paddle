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
#include <cn_api.h>
#include <cnnl.h>
#include <concurrentqueue.h>

#include <string>
#include <vector>

#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/platform/device/mlu/enforce.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DataLayout = framework::DataLayout;
using DeviceContextPool = platform::DeviceContextPool;

template <typename T>
inline cnnlDataType_t ToCnnlDataType(const T& t) {
  auto type = framework::ToDataType(t);
  return ToCnnlDataType(type);
}

template <>
inline cnnlDataType_t ToCnnlDataType(const framework::proto::VarType::Type& t) {
  cnnlDataType_t type = CNNL_DTYPE_FLOAT;
  switch (t) {
    case framework::proto::VarType::FP16:
      type = CNNL_DTYPE_HALF;
      break;
    case framework::proto::VarType::FP32:
      type = CNNL_DTYPE_FLOAT;
      break;
    case framework::proto::VarType::INT8:
      type = CNNL_DTYPE_INT8;
      break;
    case framework::proto::VarType::INT32:
      type = CNNL_DTYPE_INT32;
      break;
    case framework::proto::VarType::INT64:
      type = CNNL_DTYPE_INT64;
      break;
    case framework::proto::VarType::BOOL:
      type = CNNL_DTYPE_BOOL;
      break;
    default:
      break;
  }
  return type;
}

// Converts (via narrowing) a type T value to a type U, and checks that the
// value has no value change due to the conversion.
template <typename WideT, typename NarrowT>
NarrowT CheckedNarrowing(const WideT& wide) {
  NarrowT narrow = wide;
  CHECK_EQ(narrow, wide)
      << "checked narrowing failed; values not equal post-conversion";
  return narrow;
}

cnnlDeviceType_t GetCnnlDev(int dev_ordinal);

using CnnlTensorDesc = cnnlTensorDescriptor_t;

class MLUCnnlTensorDesc {
 public:
  MLUCnnlTensorDesc() {}

  // SE_DISALLOW_COPY_AND_ASSIGN
  MLUCnnlTensorDesc(const MLUCnnlTensorDesc& desc) = delete;
  MLUCnnlTensorDesc& operator=(const MLUCnnlTensorDesc&) = delete;

  MLUCnnlTensorDesc(MLUCnnlTensorDesc&& rhs)
      : raw_tensor_desc(rhs.raw_tensor_desc) {
    rhs.raw_tensor_desc = nullptr;
  }

  MLUCnnlTensorDesc& operator=(MLUCnnlTensorDesc&& rhs);

  MLUCnnlTensorDesc(const int tensor_dim, const int dim_sizes[],
                    const cnnlDataType_t tensor_dtype);

  MLUCnnlTensorDesc(const int tensor_dim, const int dim_sizes[],
                    const cnnlDataType_t tensor_dtype,
                    const cnnlTensorLayout_t layout);

  MLUCnnlTensorDesc(const int tensor_dim, const int dim_sizes[],
                    const cnnlDataType_t tensor_dtype, int position);

  MLUCnnlTensorDesc(const int tensor_dim, const int64_t dim_sizes[],
                    const cnnlDataType_t tensor_dtype);

  MLUCnnlTensorDesc(const int tensor_dim, const int64_t dim_sizes[],
                    const cnnlDataType_t tensor_dtype,
                    const cnnlTensorLayout_t layout);

  MLUCnnlTensorDesc(const int tensor_dim, const int64_t dim_sizes[],
                    const cnnlDataType_t tensor_dtype, int position);

  MLUCnnlTensorDesc(const Tensor& tensor, const cnnlTensorLayout_t layout,
                    const cnnlDataType_t tensor_dtype);

  MLUCnnlTensorDesc(const Tensor& tensor, cnnlTensorLayout_t layout,
                    const cnnlDataType_t tensor_dtype, int position);

  MLUCnnlTensorDesc(const Tensor& tensor, cnnlTensorLayout_t layout,
                    const cnnlDataType_t tensor_dtype, int position,
                    float scale);

  ~MLUCnnlTensorDesc();

  const cnnlTensorDescriptor_t get() const { return raw_tensor_desc; }

 private:
  cnnlTensorDescriptor_t raw_tensor_desc = nullptr;
};

class MLUCnnlActivationDesc {
 public:
  MLUCnnlActivationDesc(const MLUCnnlActivationDesc& desc) = delete;
  MLUCnnlActivationDesc& operator=(const MLUCnnlActivationDesc& desc) = delete;
  MLUCnnlActivationDesc(const cnnlActivationMode_t act_mode, const float ceof);

  const cnnlActivationDescriptor_t get() const;
  ~MLUCnnlActivationDesc();

 private:
  cnnlActivationDescriptor_t active_desc_ = nullptr;
};

class MLUCnnl {
 public:
  static void Active(const platform::MLUDeviceContext& ctx,
                     cnnlActivationDescriptor_t active_desc,
                     const cnnlTensorDescriptor_t input_desc, const void* input,
                     const cnnlTensorDescriptor_t output_desc, void* output);

  static void ActiveGrad(const platform::MLUDeviceContext& ctx,
                         cnnlActivationDescriptor_t active_desc,
                         const void* alpha, const void* beta,
                         const cnnlTensorDescriptor_t y_desc, const void* y,
                         const cnnlTensorDescriptor_t diff_y_desc,
                         const void* diff_y,
                         const cnnlTensorDescriptor_t x_desc, const void* x,
                         const cnnlTensorDescriptor_t diff_x_desc,
                         void* diff_x);
};

}  // namespace operators
}  // namespace paddle
