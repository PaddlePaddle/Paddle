/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
#include <paddle/fluid/framework/operator.h>

#include <string>
#include <vector>

#include "paddle/phi/core/tensor_utils.h"

namespace paddle {
namespace operators {

template <typename T = int32_t>
inline std::vector<T> GetDataFromTensorList(
    const std::vector<const phi::DenseTensor*>& list_tensor) {
  std::vector<T> vec_new_data;
  for (size_t i = 0; i < list_tensor.size(); ++i) {
    auto tensor = list_tensor[i];
    PADDLE_ENFORCE_EQ(
        tensor->dims(),
        common::make_ddim({1}),
        phi::errors::InvalidArgument("The shape of Tensor in list must be [1]. "
                                     "But received its shape "
                                     "is [%s]",
                                     tensor->dims()));

    if (framework::TransToProtoVarType(tensor->dtype()) ==
        framework::proto::VarType::INT32) {
      if (!(tensor->place().GetType() == phi::AllocationType::CPU)) {
        phi::DenseTensor temp;
        paddle::framework::TensorCopySync(*tensor, phi::CPUPlace(), &temp);
        vec_new_data.push_back(static_cast<T>(*temp.data<int>()));
      } else {
        vec_new_data.push_back(static_cast<T>(*tensor->data<int>()));
      }
    } else if (framework::TransToProtoVarType(tensor->dtype()) ==
               framework::proto::VarType::INT64) {
      if (!(tensor->place().GetType() == phi::AllocationType::CPU)) {
        phi::DenseTensor temp;
        paddle::framework::TensorCopySync(*tensor, phi::CPUPlace(), &temp);
        // NOTE: Converting int64 to int32 may cause data overflow.
        vec_new_data.push_back(static_cast<T>(*temp.data<int64_t>()));
      } else {
        vec_new_data.push_back(static_cast<T>(*tensor->data<int64_t>()));
      }
    } else {
      PADDLE_THROW(phi::errors::InvalidArgument(
          "The dtype of Tensor in list must be int32 or int64, but received: "
          "%s",
          tensor->dtype()));
    }
  }
  return vec_new_data;
}

inline phi::DDim GetShape(const framework::ExecutionContext& ctx) {
  // 1. shape is a Tensor
  if (ctx.HasInput("ShapeTensor")) {
    auto* shape_tensor = ctx.Input<phi::DenseTensor>("ShapeTensor");
    auto vec_shape = phi::GetVectorFromTensor<int>(shape_tensor);
    return common::make_ddim(vec_shape);
  }

  // 2. shape is a list/tuple containing Tensor
  auto shape_tensor_list = ctx.MultiInput<phi::DenseTensor>("ShapeTensorList");
  if (shape_tensor_list.size() > 0) {
    auto vec_shape = GetDataFromTensorList(shape_tensor_list);
    return common::make_ddim(vec_shape);
  }

  // 3. shape is a list/tuple without containing Tensor
  auto vec_shape = ctx.Attr<std::vector<int64_t>>("shape");
  return common::make_ddim(vec_shape);
}

template <typename T>
inline T GetValue(const phi::DenseTensor* x) {
  T value = static_cast<T>(0);
  if (!(x->place().GetType() == phi::AllocationType::CPU)) {
    phi::DenseTensor cpu_x;
    framework::TensorCopy(*x, phi::CPUPlace(), &cpu_x);
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    const platform::DeviceContext* dev_ctx = pool.Get(x->place());
    dev_ctx->Wait();
#endif
    value = cpu_x.data<T>()[0];
  } else {
    value = x->data<T>()[0];
  }
  return value;
}

}  // namespace operators
}  // namespace paddle
