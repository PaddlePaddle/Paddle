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
#include "paddle/fluid/framework/tensor_util.h"

namespace paddle {
namespace operators {

// ####################   InferShape Template Function   ####################
inline bool HasCompiledContent(const framework::InferShapeContext* ctx,
                               const std::string& name) {
  bool flag = false;
  if (!ctx->IsRuntime()) {
    auto var_descs = ctx->GetInputVarPtrs(name);
    if (!var_descs.empty()) {
      flag = std::any_of(
          var_descs.begin(), var_descs.end(),
          [](framework::InferShapeVarPtr& var_ptr) {
            return BOOST_GET(framework::VarDesc*, var_ptr)->HasDescValue();
          });
    }
  }
  return flag;
}

template <typename T>
inline std::vector<T> GetDataFromVarDesc(const framework::VarDesc* var_desc) {
  PADDLE_ENFORCE_NOT_NULL(var_desc, platform::errors::Unavailable(
                                        "var_desc shall not be nullptr."));
  return framework::ExtractTensorDescValue<std::vector<T>>(
      var_desc->GetDescValue());
}

template <typename T>
inline std::vector<T> GetDataFromVarDesc(
    const framework::InferShapeContext* ctx, const std::string& attr_name) {
  auto var_descs = ctx->GetAttrVarPtrs(attr_name);
  auto var_num = var_descs.size();
  PADDLE_ENFORCE_GE(var_num, 1,
                    platform::errors::PreconditionNotMet(
                        "Required Attribute(%s) shall have at least "
                        "one VarDesc, but received %d.",
                        attr_name, var_num));

  // In case of TensorDesc<T> containing all value in single var_desc.
  if (var_num == 1) {
    auto* var_desc = BOOST_GET(framework::VarDesc*, var_descs[0]);
    return GetDataFromVarDesc<T>(var_desc);
  }

  std::vector<T> res(var_num);
  // In case of list[TensorDesc<T>] and each of them only contains one value.
  for (size_t i = 0; i < var_num; ++i) {
    auto* var_desc = BOOST_GET(framework::VarDesc*, var_descs[i]);
    auto vals = GetDataFromVarDesc<T>(var_desc);
    PADDLE_ENFORCE_EQ(vals.size(), 1,
                      platform::errors::PreconditionNotMet(
                          "%s shall have only one value, but got %d.",
                          var_desc->Name(), vals.size()));
    res[i] = vals[0];
  }
  return res;
}

template <typename T>
inline T GetScalarDataFromVarDesc(const framework::InferShapeContext* ctx,
                                  const std::string& attr_name) {
  auto content = GetDataFromVarDesc<T>(ctx, attr_name);
  PADDLE_ENFORCE_EQ(
      content.size(), 1,
      platform::errors::PreconditionNotMet(
          "Required content.size() == 1, but received %d.", content.size()));
  return content[0];
}

template <typename T>
std::vector<T> GetScalars(const framework::InferShapeContext* ctx,
                          const std::string& attr_name) {
  if (ctx->HasAttrVar(attr_name)) {
    return GetDataFromVarDesc<T>(ctx, attr_name);
  } else {
    return ctx->Attrs().Get<std::vector<T>>(attr_name);
  }
}

template <typename T>
T GetScalar(const framework::InferShapeContext* ctx,
            const std::string& attr_name) {
  auto res = GetScalars<T>(ctx, attr_name);
  PADDLE_ENFORCE_EQ(
      res.size(), 1,
      platform::errors::PreconditionNotMet(
          "Required content.size() == 1, but received %d.", res.size()));
  return res[0];
}

// ####################   Execution Template Function    ####################
template <typename T>
inline std::vector<T> GetDataFromVariable(const framework::Variable* var) {
  PADDLE_ENFORCE_NOT_NULL(
      var, platform::errors::Unavailable("var shall not be nullptr."));
  PADDLE_ENFORCE_EQ(
      var->IsType<framework::LoDTensor>(), true,
      platform::errors::PreconditionNotMet(
          "Only support to parse data with LoDTensor, but received %s.",
          platform::demangle(framework::ToTypeName(var->Type()))));

  auto& tensor = var->Get<framework::LoDTensor>();
  auto place = tensor.place();
  PADDLE_ENFORCE_EQ(platform::is_cpu_place(place), true,
                    platform::errors::PreconditionNotMet(
                        "Tensor should be in CPUPlace, but got %s", place));
  PADDLE_ENFORCE_EQ(
      tensor.dims().size(), 1,
      platform::errors::PreconditionNotMet(
          "Rank of Tensor should be 1 , but got DDim: %s", tensor.dims()));

  std::vector<T> res;
  framework::TensorToVector(tensor, &res);

  return res;
}

template <typename T>
inline std::vector<T> GetDataFromVariable(
    const std::vector<framework::Variable*> vars,
    const std::string& attr_name) {
  auto var_num = vars.size();
  PADDLE_ENFORCE_GE(var_num, 1,
                    platform::errors::PreconditionNotMet(
                        "Required Attribute(%s) shall have at least "
                        "one Variable, but received %d.",
                        attr_name, var_num));

  // In case of containing all values in single Variable.
  if (var_num == 1) {
    return GetDataFromVariable<T>(vars[0]);
  }
  std::vector<T> res(var_num);
  // In case of List[Variable] and each of them only contains one value.
  for (size_t i = 0; i < var_num; ++i) {
    auto vals = GetDataFromVariable<T>(vars[i]);
    PADDLE_ENFORCE_EQ(vals.size(), 1,
                      platform::errors::PreconditionNotMet(
                          "%s[%d] shall have only one value, but got %d.",
                          attr_name, i, vals.size()));
    res[i] = vals[0];
  }
  return res;
}

template <typename T>
inline std::vector<T> GetDataFromVariable(
    const framework::ExecutionContext& ctx, const std::string& attr_name) {
  auto vars = ctx.MultiAttrVar(attr_name);
  return GetDataFromVariable<T>(vars, attr_name);
}

template <typename T>
std::vector<T> GetScalars(const framework::ExecutionContext& ctx,
                          const std::string& attr_name) {
  if (ctx.HasAttrVar(attr_name)) {
    return GetDataFromVariable<T>(ctx, attr_name);
  } else {
    return ctx.Attr<std::vector<T>>(attr_name);
  }
}

template <typename T>
T GetScalar(const framework::ExecutionContext& ctx,
            const std::string& attr_name) {
  auto res = GetScalars<T>(ctx, attr_name);
  PADDLE_ENFORCE_EQ(
      res.size(), 1,
      platform::errors::PreconditionNotMet(
          "Required content.size() == 1, but received %d.", res.size()));
  return res[0];
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
