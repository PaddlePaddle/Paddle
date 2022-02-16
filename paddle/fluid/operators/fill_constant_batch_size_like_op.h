/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/pten/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class FillConstantBatchSizeLikeOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto data_type =
        static_cast<framework::proto::VarType::Type>(ctx.Attr<int>("dtype"));
    auto float_value = ctx.Attr<float>("value");
    auto str_value = ctx.Attr<std::string>("str_value");
    auto force_cpu = ctx.Attr<bool>("force_cpu");

    auto *out = ctx.Output<framework::Tensor>("Out");
    auto *in = ctx.Input<framework::LoDTensor>("Input");
    if (in->lod().size() && ctx.Attr<int>("input_dim_idx") == 0) {
      // set the correct batch size for the LoDTensor.
      auto odims = out->dims();
      int output_dim_idx = ctx.Attr<int>("output_dim_idx");
      odims[output_dim_idx] = static_cast<int>(in->lod().back().size()) - 1;
      out->mutable_data<T>(odims, ctx.GetPlace());
    }

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

    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    bool cpu_place = force_cpu || ctx.GetPlace() == platform::CPUPlace();
    if (cpu_place) {
      auto &dev_ctx = *pool.Get(platform::CPUPlace());
      pten::funcs::SetConstant<platform::CPUDeviceContext, T> functor;
      out->mutable_data(platform::CPUPlace(),
                        framework::TransToPtenDataType(data_type));
      functor(reinterpret_cast<const platform::CPUDeviceContext &>(dev_ctx),
              out, static_cast<T>(value));
    }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    if (!cpu_place) {
      auto &dev_ctx = *pool.Get(ctx.GetPlace());
      pten::funcs::SetConstant<platform::CUDADeviceContext, T> functor;
      out->mutable_data(ctx.GetPlace(),
                        framework::TransToPtenDataType(data_type));
      functor(reinterpret_cast<const platform::CUDADeviceContext &>(dev_ctx),
              out, static_cast<T>(value));
    }
#endif
  }
};

}  // namespace operators
}  // namespace paddle
