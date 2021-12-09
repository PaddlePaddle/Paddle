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
#include <string>
#include <vector>
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/shape_inference.h"
#include "paddle/fluid/framework/tensor_desc.h"

namespace paddle {
namespace operators {

// TODO(Aurelius84): Make it into template function
template <typename T>
inline std::vector<T> GetDataFromVarDesc(const framework::VarDesc* var_desc) {
  PADDLE_ENFORCE_NOT_NULL(var_desc, platform::errors::Unavailable(
                                        "var_desc shall not be nullptr."));
  return framework::ExtractTensorDescValue<std::vector<T>>(
      var_desc->GetDescValue());
}

template <typename T>
inline std::vector<T> GetVecDataFromVarDesc(framework::InferShapeContext* ctx,
                                            const std::string& name) {
  auto var_descs = ctx->GetInputVarPtrs(name);
  PADDLE_ENFORCE_EQ(
      var_descs.size(), 1,
      platform::errors::PreconditionNotMet(
          "Required Input(%s) has only one VarDesc, but received %d.", name,
          var_descs.size()));
  auto* var_desc = BOOST_GET(framework::VarDesc*, var_descs[0]);
  return GetDataFromVarDesc<T>(var_desc);
}

template <typename T>
inline T GetScalarDataFromVarDesc(framework::InferShapeContext* ctx,
                                  const std::string& name) {
  auto content = GetVecDataFromVarDesc<T>(ctx, name);
  PADDLE_ENFORCE_EQ(
      content.size(), 1,
      platform::errors::PreconditionNotMet(
          "Required content.size() == 1, but received %d.", content.size()));
  return content[0];
}

template <typename T>
inline std::vector<T> GetScalarsFromVarDescs(framework::InferShapeContext* ctx,
                                             const std::string& name,
                                             T default_val) {
  auto var_descs = ctx->GetInputVarPtrs(name);
  std::vector<T> scalar_vals(var_descs.size(), default_val);
  for (size_t i = 0; i < var_descs.size(); ++i) {
    auto* var_desc = BOOST_GET(framework::VarDesc*, var_descs[i]);
    auto scalar = GetDataFromVarDesc<T>(var_desc);
    if (!scalar.empty()) {
      scalar_vals[i] = scalar[0];
    }
  }
  return scalar_vals;
}

inline bool HasCompiledContent(const framework::InferShapeContext* ctx,
                               const std::string& name) {
  bool flag = false;
  if (!ctx->IsRuntime() && ctx->HasInput(name)) {
    auto var_descs = ctx->GetInputVarPtrs(name);
    if (!var_descs.empty()) {
      flag = std::any_of(var_descs.begin(), var_descs.end(),
                         [](framework::InferShapeVarPtr& var_ptr) {
                           // FIXME(Aurelius84): return
                           // BOOST_GET(framework::VarDesc*,
                           // var_ptr)->HasContent();
                           return true;
                         });
    }
  }
  return flag;
}

template <typename T = int32_t>
inline std::vector<T> GetDataFromTensor(const framework::Tensor* x) {
  std::vector<T> vec_new_data;
  if (x->type() == framework::proto::VarType::INT32) {
    auto* data = x->data<int>();
    framework::Tensor cpu_attr_tensor;
    if (!platform::is_cpu_place(x->place())) {
      TensorCopySync(*x, platform::CPUPlace(), &cpu_attr_tensor);
      data = cpu_attr_tensor.data<int>();
    }
    vec_new_data = std::vector<T>(data, data + x->numel());
  } else if (x->type() == framework::proto::VarType::INT64) {
    auto* data = x->data<int64_t>();
    framework::Tensor cpu_attr_tensor;
    if (!platform::is_cpu_place(x->place())) {
      TensorCopySync(*x, platform::CPUPlace(), &cpu_attr_tensor);
      data = cpu_attr_tensor.data<int64_t>();
    }
    // NOTE: Converting int64 to int32 may cause data overflow.
    vec_new_data = std::vector<T>(data, data + x->numel());
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "The dtype of Tensor must be int32 or int64, but received: %s",
        x->type()));
  }
  return vec_new_data;
}

template <typename T = int32_t>
inline std::vector<T> GetDataFromTensorList(
    const std::vector<const framework::Tensor*>& list_tensor) {
  std::vector<T> vec_new_data;
  for (size_t i = 0; i < list_tensor.size(); ++i) {
    auto tensor = list_tensor[i];
    PADDLE_ENFORCE_EQ(tensor->dims(), framework::make_ddim({1}),
                      platform::errors::InvalidArgument(
                          "The shape of Tensor in list must be [1]. "
                          "But received its shape "
                          "is [%s]",
                          tensor->dims()));

    if (tensor->type() == framework::proto::VarType::INT32) {
      if (!platform::is_cpu_place(tensor->place())) {
        framework::Tensor temp;
        TensorCopySync(*tensor, platform::CPUPlace(), &temp);
        vec_new_data.push_back(static_cast<T>(*temp.data<int>()));
      } else {
        vec_new_data.push_back(static_cast<T>(*tensor->data<int>()));
      }
    } else if (tensor->type() == framework::proto::VarType::INT64) {
      if (!platform::is_cpu_place(tensor->place())) {
        framework::Tensor temp;
        TensorCopySync(*tensor, platform::CPUPlace(), &temp);
        // NOTE: Converting int64 to int32 may cause data overflow.
        vec_new_data.push_back(static_cast<T>(*temp.data<int64_t>()));
      } else {
        vec_new_data.push_back(static_cast<T>(*tensor->data<int64_t>()));
      }
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "The dtype of Tensor in list must be int32 or int64, but received: "
          "%s",
          tensor->type()));
    }
  }
  return vec_new_data;
}

inline framework::DDim GetShape(const framework::ExecutionContext& ctx) {
  // 1. shape is a Tensor
  if (ctx.HasInput("ShapeTensor")) {
    auto* shape_tensor = ctx.Input<framework::LoDTensor>("ShapeTensor");
    auto vec_shape = GetDataFromTensor<int>(shape_tensor);
    return framework::make_ddim(vec_shape);
  }

  // 2. shape is a list/tuple containing Tensor
  auto shape_tensor_list = ctx.MultiInput<framework::Tensor>("ShapeTensorList");
  if (shape_tensor_list.size() > 0) {
    auto vec_shape = GetDataFromTensorList(shape_tensor_list);
    return framework::make_ddim(vec_shape);
  }

  // 3. shape is a list/tuple without containing Tensor
  auto vec_shape = ctx.Attr<std::vector<int64_t>>("shape");
  return framework::make_ddim(vec_shape);
}

template <typename T>
inline T GetValue(const framework::Tensor* x) {
  T value = static_cast<T>(0);
  if (!platform::is_cpu_place(x->place())) {
    framework::Tensor cpu_x;
    framework::TensorCopy(*x, platform::CPUPlace(), &cpu_x);
#ifdef PADDLE_WITH_ASCEND_CL
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
