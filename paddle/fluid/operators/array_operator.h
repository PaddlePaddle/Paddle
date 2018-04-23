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
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace operators {
class ArrayOp : public framework::OperatorBase {
 public:
  ArrayOp(const std::string &type, const framework::VariableNameMap &inputs,
          const framework::VariableNameMap &outputs,
          const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 protected:
  size_t GetOffset(const framework::Scope &scope,
                   const platform::Place &place) const {
    auto *i = scope.FindVar(Input("I"));
    PADDLE_ENFORCE(i != nullptr, "I must be set");
    auto &i_tensor = i->Get<framework::LoDTensor>();
    PADDLE_ENFORCE_EQ(i_tensor.numel(), 1);

    // get device context from pool
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto &dev_ctx = *pool.Get(place);

    size_t offset;
    if (platform::is_gpu_place(i_tensor.place())) {
      // FIXME: Avoid copy from GPU to CPU
      framework::Tensor t;
      framework::TensorCopy(i_tensor, platform::CPUPlace(), dev_ctx, &t);
      dev_ctx.Wait();
      offset = static_cast<size_t>(*t.data<int64_t>());
    } else {
      offset = static_cast<size_t>(*i_tensor.data<int64_t>());
    }
    VLOG(10) << " Offset = " << offset;
    return offset;
  }
};

}  // namespace operators
}  // namespace paddle
