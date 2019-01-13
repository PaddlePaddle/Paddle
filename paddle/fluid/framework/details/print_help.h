//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include <sstream>  // std::stringstream
#include <string>

// #include "paddle/fluid/framework/details/threaded_ssa_graph_executor.h"
// #include "paddle/fluid/framework/details/multi_devices_helper.h"
// #include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/details/op_handle_base.h"
#include "paddle/fluid/framework/var_type.h"
// #include "paddle/fluid/platform/profiler.h"
// #include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace framework {
namespace details {

template <typename T>
std::string GetReadableData(const void *in_data, int64_t len) {
  if (len < 0) {
    return "";
  }

  T *data = reinterpret_cast<T *>(in_data);

  std::stringstream ss;
  int64_t r0 = 0;
  for (r0 = 0; r0 < len && r0 < 20; r0++) {
    ss << data[r0] << ",";
  }

  int64_t r1 = len - 20;
  if (r1 <= r0) {
    r1 = r0;
  } else {
    ss << "...";
  }

  for (; r1 < len; r1++) {
    ss << data[r1] << ",";
  }

  return ss.str();
}

inline std::string GetTensorInfo(const framework::Tensor &in_tensor) {
  std::stringstream ss;
  if (!in_tensor.IsInitialized()) {
    ss << " not initialized";
    return ss.str();
  }

  ss << " place:" << in_tensor.place() << ", dims:[" << in_tensor.dims() << "]";

  auto dtype = framework::ToTypeIndex(in_tensor.type());
  framework::Tensor tensor;
  if (platform::is_cpu_place(in_tensor.place())) {
    tensor.ShareDataWith(in_tensor);
  } else {
    // copy data to cpu to print
    platform::CPUPlace cpu_place;
    framework::TensorCopy(in_tensor, cpu_place, &tensor);
  }

  ss << ", data:[";
  void *data = tensor.data<void>();
  if (framework::IsType<const float>(dtype)) {
    ss << GetReadableData<const float>(data, tensor.numel());
  } else if (framework::IsType<const double>(dtype)) {
    ss << GetReadableData<const double>(data, tensor.numel());
  } else if (framework::IsType<const int>(dtype)) {
    ss << GetReadableData<const int>(data, tensor.numel());
  } else if (framework::IsType<const int64_t>(dtype)) {
    ss << GetReadableData<const int64_t>(data, tensor.numel());
  } else if (framework::IsType<const bool>(dtype)) {
    ss << GetReadableData<const bool>(data, tensor.numel());
  } else {
    // TODO(gongwb): add more data types support.
    ss << "\tdata: unprintable type: " << dtype.name();
  }
  ss << "]";

  return ss.str();
}

inline std::string GetSelectedRowsInfo(const framework::SelectedRows &slr) {
  std::stringstream ss;
  ss << "height:" << slr.height() << ", rows:[";
  for (unsigned int i = 0; i < slr.rows().size(); i++) {
    if (i != slr.rows().size() - 1) {
      ss << slr.rows()[i] << ",";
    } else {
      ss << slr.rows()[i];
    }
  }
  ss << "], tensor:" << GetTensorInfo(slr.value());

  return ss.str();
}

inline std::string GetVarInfo_(framework::Scope *local_scope,
                               const std::string &name) {
  auto var = local_scope->FindVar(name);

  std::stringstream ss;
  if (var == NULL) {
    ss << "can't find " << name
       << GenScopeTreeDebugInfo(const_cast<framework::Scope *>(local_scope));
    return ss.str();
  }

  if (var->IsType<framework::LoDTensor>()) {
    return GetTensorInfo(var->Get<LoDTensor>());
  }

  if (var->IsType<framework::SelectedRows>()) {
    return GetSelectedRowsInfo(var->Get<SelectedRows>());
  }

  ss << "can't print " << name;
  return ss.str();
}

inline std::string GetVarInfo(framework::Scope *scope,
                              const std::string &name) {
  framework::Scope *local_scope =
      scope->FindVar(kLocalExecScopeName)->Get<framework::Scope *>();
  return GetVarInfo_(local_scope, name);
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
