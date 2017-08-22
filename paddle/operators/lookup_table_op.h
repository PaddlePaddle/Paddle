/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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

#include "paddle/framework/op_registry.h"
#include "paddle/operators/functor/math_functor.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
class LookupTableKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto table_t = context.Input<Tensor>("W");      // float tensor
    auto ids_t = context.Input<Tensor>("Ids");      // int tensor
    auto output_t = context.Output<Tensor>("Out");  // float tensor

    size_t N = table_t->dims()[0];
    size_t D = table_t->dims()[1];
    auto ids = ids_t->data<int32_t>();
    auto table = table_t->data<T>();
    auto output = output_t->mutable_data<T>(context.GetPlace());
    for (size_t i = 0; i < product(ids_t->dims()); ++i) {
      PADDLE_ENFORCE_LT(ids[i], N);
      PADDLE_ENFORCE_GE(ids[i], 0);
      memcpy(output + i * D, table + ids[i] * D, D * sizeof(T));
    }
  }
};

template <typename T>
class LookupTableGradKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto ids_t = context.Input<Tensor>("Ids");
    auto d_output_t = context.Input<Tensor>(framework::GradVarName("Out"));
    auto d_table_t = context.Output<Tensor>(framework::GradVarName("W"));

    size_t N = d_table_t->dims()[0];
    size_t D = d_table_t->dims()[1];
    auto ids = ids_t->data<int32_t>();
    const T* d_output = d_output_t->data<T>();
    T* d_table = d_table_t->mutable_data<T>(context.GetPlace());

    auto* device_context =
        const_cast<platform::DeviceContext*>(context.device_context_);
    functor::Set<paddle::platform::CPUPlace, T>()(static_cast<T>(0), d_table_t,
                                                  device_context);
    for (size_t i = 0; i < product(ids_t->dims()); ++i) {
      PADDLE_ENFORCE_LT(ids[i], N);
      PADDLE_ENFORCE_GE(ids[i], 0);
      for (size_t j = 0; j < D; ++j) {
        d_table[ids[i] * D + j] += d_output[i * D + j];
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
