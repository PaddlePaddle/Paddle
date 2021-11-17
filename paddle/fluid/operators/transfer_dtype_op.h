// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/data_transform.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/data_type_transform.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace platform {
class DeviceContext;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace framework {
class LoDTensor;
class Variable;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace operators {
using DataLayout = framework::DataLayout;

class TransferDtypeFunctor {
 public:
  TransferDtypeFunctor(const framework::Variable *in, framework::Variable *out,
                       const platform::DeviceContext &dev_ctx,
                       const int dst_dtype)
      : in_(in), out_(out), dev_ctx_(dev_ctx), dst_dtype_(dst_dtype) {}

  void operator()() const {
    auto &in_tensor = *framework::GetLoDTensorOrSelectedRowsValueFromVar(*in_);

    framework::LoDTensor out_tensor;

    auto out_dtype = static_cast<framework::proto::VarType::Type>(dst_dtype_);
    framework::TransDataType(in_tensor, out_dtype, &out_tensor);
    framework::SetTensorToVariable(*in_, out_tensor, out_);
  }

 private:
  const framework::Variable *in_;
  framework::Variable *out_;
  const platform::DeviceContext &dev_ctx_;
  const int dst_dtype_;
};

}  // namespace operators
}  // namespace paddle
