// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <sstream>  // std::stringstream
#include <string>

#include "paddle/fluid/framework/details/print_utils.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/framework/variable.h"

namespace paddle {
namespace framework {
namespace details {

std::string GetVariableReadableData(const framework::Tensor* input_tensor,
                                    const std::string& message,
                                    const std::string& name) {
  std::stringstream ss;
  auto& in_tensor = *input_tensor;
  if (!in_tensor.IsInitialized()) {
    ss << message << " " << name << " onit initialized";
    return ss.str();
  }

  auto dtype = in_tensor.type();
  size_t size = in_tensor.numel();
  size_t summarize = 10;

  framework::Tensor tensor;
  // tensor.set_lod(tensor.lod());
  tensor.Resize(tensor.dims());

  if (platform::is_cpu_place(in_tensor.place())) {
    tensor.ShareDataWith(in_tensor);
  } else {
    // copy data to cpu to print
    platform::CPUPlace cpu_place;
    framework::TensorCopy(in_tensor, cpu_place, &tensor);
  }
  void* data = const_cast<void*>(tensor.data<void>());

  ss << message << " " << name << ":";
  if (IsType<const float>(dtype)) {
    ss << GetArrayReadableData<float>(0, 1, size, summarize, data);
  } else if (IsType<const double>(dtype)) {
    ss << GetArrayReadableData<double>(0, 1, size, summarize, data);
  } else if (IsType<const int>(dtype)) {
    ss << GetArrayReadableData<int>(0, 1, size, summarize, data);
  } else if (IsType<const int64_t>(dtype)) {
    ss << GetArrayReadableData<int64_t>(0, 1, size, summarize, data);
  } else if (IsType<const bool>(dtype)) {
    ss << GetArrayReadableData<bool>(0, 1, size, summarize, data);
  } else {
    // TODO(gongwb): add more data types support.
    ss << "\tdata: unprintable type: " << dtype.name();
  }

  return ss.str();
}

std::string GetVariableReadableData(const Scope& scope,
                                    const std::string& message,
                                    const std::string& name) {
  std::stringstream ss;

  auto var = scope.FindVar(name);
  if (var == NULL) {
    ss << "can't find " << message << " " << name;
    return ss.str();
  }

  if (var->IsType<framework::SelectedRows>()) {
    return GetVariableReadableData(
        &var->GetMutable<framework::SelectedRows>()->value(), message, name);
  } else if (var->IsType<framework::LoDTensor>()) {
    return GetVariableReadableData(var->GetMutable<framework::LoDTensor>(),
                                   message, name);
  } else {
    PADDLE_ENFORCE(false, "only support SelectedRows and lodTensor");
  }

  return "";
}
}  // namespace details
}  // namespace framework
}  // namespace paddle
