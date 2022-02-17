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
#include <functional>
#include "paddle/fluid/framework/data_type_transform.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/pten/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
class CPULinspaceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* pre_start = context.Input<framework::Tensor>("Start");
    auto* pre_stop = context.Input<framework::Tensor>("Stop");
    int32_t num = context.Input<framework::Tensor>("Num")->data<int32_t>()[0];
    auto* out = context.Output<framework::Tensor>("Out");
    auto dtype = static_cast<framework::proto::VarType::Type>(
        context.Attr<int>("dtype"));

    Tensor start_t;
    Tensor stop_t;
    auto start_dtype = framework::OpKernelType(
        framework::TransToProtoVarType(pre_start->dtype()), context.GetPlace());
    auto stop_dtype = framework::OpKernelType(
        framework::TransToProtoVarType(pre_stop->dtype()), context.GetPlace());
    auto out_dtype = framework::OpKernelType(dtype, context.GetPlace());
    framework::TransDataType(start_dtype, out_dtype, *pre_start, &start_t);
    framework::TransDataType(stop_dtype, out_dtype, *pre_stop, &stop_t);

    T start = start_t.data<T>()[0];
    T stop = stop_t.data<T>()[0];
    PADDLE_ENFORCE_GT(num, 0, platform::errors::InvalidArgument(
                                  "The num of linspace op should be larger "
                                  "than 0, but received num is %d",
                                  num));

    out->Resize(framework::make_ddim({num}));

    T* out_data = out->mutable_data<T>(context.GetPlace());

    if (num > 1) {
      // step should be of double type for all types
      double step = (static_cast<double>(stop - start)) / (num - 1);
      int half_num = num / 2;
      for (int i = 0; i < num; ++i) {
        if (i < half_num) {
          out_data[i] = static_cast<T>(start + step * i);
        } else {
          out_data[i] = static_cast<T>(stop - step * (num - i - 1));
        }
      }
    } else {
      out_data[0] = static_cast<T>(start);
    }
  }
};

}  // namespace operators
}  // namespace paddle
