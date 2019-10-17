// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/platform/device_memory_aligment.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class SyncFusedTensorOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto &in_var_names = context.Inputs("X");
    auto &out_var_name = context.Outputs("Out");
    auto &in_vars = context.MultiInputVar("X");
    auto *out_var = context.OutputVar("Out");
    bool out_hold = context.Attr<bool>("out_hold");

    // Variable type check
    for (size_t i = 0; i < in_var_names.size(); ++i) {
      PADDLE_ENFORCE_EQ(
          in_vars[i]->IsInitialized(), true,
          "The Input Variable X(%s) of SyncFusedTensorOp is not initialized.",
          in_var_names[i]);
      PADDLE_ENFORCE_EQ(in_vars[i]->IsType<framework::LoDTensor>(), true,
                        "SyncFusedTensorOp only support LoDTensor input.");
    }
    PADDLE_ENFORCE_EQ(
        out_var->IsInitialized(), true,
        "The Output Variable Out(%s) of SyncFusedTensorOp is not initialized.",
        out_var_name.front());
    PADDLE_ENFORCE_EQ(out_var->IsType<framework::LoDTensor>(), true,
                      "SyncFusedTensorOp only esupport LoDTensor output.");

    auto in_tensors = context.MultiInput<framework::LoDTensor>("X");
    auto out_tensor = context.Output<framework::LoDTensor>("Out");

    // Input tensor data check
    for (size_t i = 0; i < in_tensors.size(); ++i) {
      PADDLE_ENFORCE_EQ(in_tensors[i]->IsInitialized(), true,
                        "The Tensor in Input Variable X(%s) of "
                        "SyncFusedTensorOp is not initialized.",
                        in_var_names[i]);
    }

    // Output tensor init or check
    auto dtype = in_tensors.at(0)->type();
    size_t size_of_dtype = framework::SizeOfType(dtype);
    if (out_hold) {
      PADDLE_ENFORCE_EQ(out_tensor->IsInitialized(), true,
                        "The Tensor in Output Variable Out(%s) of "
                        "SyncFusedTensorOp is not initialized. "
                        "SyncFusedTensorOp needs output is also initialized.",
                        out_var_name.front());
    } else {
      out_tensor
          ->Resize(
              framework::make_ddim({static_cast<int64_t>(GetFusedMemorySize(
                  in_tensors, size_of_dtype, context.GetPlace()))}))
          .mutable_data(context.GetPlace(), dtype);
    }

    size_t offset = 0;
    for (size_t i = 0; i < in_tensors.size(); ++i) {
      size_t len = static_cast<size_t>(in_tensors[i]->numel());
      auto sub_tensor = out_tensor->Slice(static_cast<int64_t>(offset),
                                          static_cast<int64_t>(offset + len));
      if (out_hold) {
        if (!is_same_place(in_tensors[i]->place(), out_tensor->place())) {
          framework::TensorCopy(*in_tensors[i], context.GetPlace(),
                                &sub_tensor);
        }
      } else {
        framework::TensorCopy(*in_tensors[i], context.GetPlace(), &sub_tensor);
      }
      offset += platform::Alignment(len * size_of_dtype, context.GetPlace()) /
                size_of_dtype;
    }
  }

 private:
  size_t GetFusedMemorySize(
      const std::vector<const framework::LoDTensor *> &in_tensors,
      const size_t &size_of_dtype, const platform::Place &place) const {
    size_t numel = 0;
    for (size_t i = 0; i < in_tensors.size(); ++i) {
      auto size = in_tensors[i]->numel();
      PADDLE_ENFORCE_GT(size, 0, "The input tensor is empty.");
      numel += platform::Alignment(static_cast<size_t>(size) * size_of_dtype,
                                   place) /
               size_of_dtype;
    }
    return numel;
  }
};

}  // namespace operators
}  // namespace paddle
