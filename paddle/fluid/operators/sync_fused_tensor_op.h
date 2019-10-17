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
    auto &in_var_names = context.Inputs("Input");
    auto &fused_in_var_name = context.Inputs("FusedInput");
    auto &in_vars = context.MultiInputVar("Input");
    auto *fused_in_var = context.InputVar("FusedInput");

    // Variable init and type check
    for (size_t i = 0; i < in_var_names.size(); ++i) {
      PADDLE_ENFORCE_EQ(in_vars[i]->IsInitialized(), true,
                        "The Input Variable Input(%s) of SyncFusedTensorOp is "
                        "not initialized.",
                        in_var_names[i]);
      PADDLE_ENFORCE_EQ(in_vars[i]->IsType<framework::LoDTensor>(), true,
                        "SyncFusedTensorOp only support LoDTensor input.");
    }
    // PADDLE_ENFORCE_EQ(fused_in_var, fused_out_var,
    //   "The FusedInput(%s) and FusedOutout(%s) in SyncFusedTensorOp should be
    //   the same variable.", fused_in_var_name.front(),
    //   fused_out_var_name.front());
    PADDLE_ENFORCE_EQ(fused_in_var->IsInitialized(), true,
                      "The Input Variable FusedInput(%s) of SyncFusedTensorOp "
                      "is not initialized.",
                      fused_in_var_name.front());
    PADDLE_ENFORCE_EQ(fused_in_var->IsType<framework::LoDTensor>(), true,
                      "SyncFusedTensorOp only support LoDTensor input.");

    auto in_tensors = context.MultiInput<framework::LoDTensor>("Input");
    auto fused_in_tensor = context.Input<framework::LoDTensor>("FusedInput");
    auto fused_out_tensor = context.Output<framework::LoDTensor>("FusedOutput");

    // Input tensor data check
    for (size_t i = 0; i < in_tensors.size(); ++i) {
      PADDLE_ENFORCE_EQ(in_tensors[i]->IsInitialized(), true,
                        "The Tensor in Input Variable Input(%s) of "
                        "SyncFusedTensorOp is not initialized.",
                        in_var_names[i]);
    }
    PADDLE_ENFORCE_EQ(fused_in_tensor->IsInitialized(), true,
                      "The Tensor in Input Variable FusedInput(%s) of "
                      "SyncFusedTensorOp is not initialized. ",
                      fused_in_var_name.front());

    // Data check and copy
    auto dtype = in_tensors.at(0)->type();
    size_t size_of_dtype = framework::SizeOfType(dtype);
    size_t offset = 0;
    for (size_t i = 0; i < in_tensors.size(); ++i) {
      size_t len = static_cast<size_t>(in_tensors[i]->numel());
      auto sub_tensor = fused_in_tensor->Slice(
          static_cast<int64_t>(offset), static_cast<int64_t>(offset + len));
      if (!IsSameTensor(*in_tensors[i], sub_tensor)) {
        framework::TensorCopy(*in_tensors[i], context.GetPlace(), &sub_tensor);
      }
      offset += platform::Alignment(len * size_of_dtype, context.GetPlace()) /
                size_of_dtype;
    }

    // Set output, for unit test
    if (!fused_out_tensor->IsInitialized()) {
      fused_out_tensor->ShareDataWith(*fused_in_tensor);
    }
  }

 private:
  bool IsSameTensor(const framework::Tensor &x,
                    const framework::Tensor &y) const {
    return is_same_place(x.place(), y.place()) && x.type() == y.type() &&
           (x.data<T>() == y.data<T>());
  }
};

}  // namespace operators
}  // namespace paddle
