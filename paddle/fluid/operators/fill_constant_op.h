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

#include <sstream>
#include <string>
#include <vector>
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/utils.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

inline framework::DDim GetShape(const framework::ExecutionContext &ctx,
                                std::string op_type) {
  // 1. shape is a Tensor
  if (ctx.HasInput("ShapeTensor")) {
    auto *shape_tensor = ctx.Input<framework::LoDTensor>("ShapeTensor");
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
class FillConstantKernel : public framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext &ctx) const override {
    auto data_type =
        static_cast<framework::proto::VarType::Type>(ctx.Attr<int>("dtype"));

    auto str_value = ctx.Attr<std::string>("str_value");
    auto float_value = ctx.Attr<float>("value");
    auto force_cpu = ctx.Attr<bool>("force_cpu");
    framework::Tensor *tensor = nullptr;

    framework::Variable *out_var = ctx.OutputVar("Out");

    T value;
    if (str_value.empty()) {
      value = static_cast<T>(float_value);
    } else {
      std::stringstream convert_stream(str_value);
      if (std::is_same<int64_t, T>::value) {
        int64_t tmp_value;
        convert_stream >> tmp_value;
        value = static_cast<T>(tmp_value);
      } else {
        double tmp_value;
        convert_stream >> tmp_value;
        value = static_cast<T>(tmp_value);
      }
    }
    if (ctx.HasInput("ValueTensor")) {
      auto *value_tensor = ctx.Input<framework::Tensor>("ValueTensor");
      PADDLE_ENFORCE_EQ(
          value_tensor->numel(), 1,
          platform::errors::InvalidArgument(
              "When use Tensor as value to set Tensor value in fill_cosntant, "
              "value input(ValueTensor) size must be 1, but get %d",
              value_tensor->numel()));
      const T *tensor_data = value_tensor->data<T>();
      framework::Tensor cpu_tensor;
      if (platform::is_gpu_place(value_tensor->place())) {
        TensorCopySync(*value_tensor, platform::CPUPlace(), &cpu_tensor);
        tensor_data = cpu_tensor.data<T>();
      }
      value = tensor_data[0];
    }
    const std::string op_type = "fill_constant";
    auto shape = GetShape(ctx, op_type);

    if (out_var->IsType<framework::LoDTensor>()) {
      tensor = out_var->GetMutable<framework::LoDTensor>();
      tensor->Resize(shape);
    } else if (out_var->IsType<framework::SelectedRows>()) {
      tensor = out_var->GetMutable<framework::SelectedRows>()->mutable_value();
      tensor->Resize(shape);
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "In fill constant Op, the output only supports SelectedRows and "
          "LoDTensor."));
    }

    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto &dev_ctx = *pool.Get(ctx.GetPlace());
    bool cpu_place = force_cpu || ctx.GetPlace() == platform::CPUPlace();
    if (cpu_place) {
      tensor->mutable_data(platform::CPUPlace(), data_type);
      math::SetConstant<platform::CPUDeviceContext, T> functor;
      functor(reinterpret_cast<const platform::CPUDeviceContext &>(dev_ctx),
              tensor, static_cast<T>(value));
    }
#ifdef PADDLE_WITH_CUDA
    if (!cpu_place) {
      tensor->mutable_data(ctx.GetPlace(), data_type);
      math::SetConstant<platform::CUDADeviceContext, T> functor;
      functor(reinterpret_cast<const platform::CUDADeviceContext &>(dev_ctx),
              tensor, static_cast<T>(value));
    }
#endif
  }
};
}  // namespace operators
}  // namespace paddle
