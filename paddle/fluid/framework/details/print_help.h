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
#include <vector>

// #include "paddle/fluid/framework/details/threaded_ssa_graph_executor.h"
// #include "paddle/fluid/framework/details/multi_devices_helper.h"
// #include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/details/op_handle_base.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/cuda_device_guard.h"
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

inline bool TestSetValue(const platform::CUDADeviceContext &context) {
  cudaError_t e_sync = cudaStreamSynchronize(context.stream());
  if (e_sync != 0) {
    VLOG(10) << "cudaStreamSynchronize " << cudaGetErrorString(e_sync);
  }

  cudaError_t e_get = cudaGetLastError();
  if (e_get != 0) {
    VLOG(10) << "cudaGetLastError  " << cudaGetErrorString(e_get)
             << " errno:" << e_get;
    return false;
  }

  return true;
}

inline bool TestStream(const platform::CUDADeviceContext &context, int size) {
  // for(int i=0;i<=1;i++){
  auto dev_id =
      boost::get<platform::CUDAPlace>(context.GetPlace()).GetDeviceId();
  platform::CUDADeviceGuard guard(dev_id);
  void *ptr;
  auto status = cudaMalloc(&ptr, size);
  if (UNLIKELY(status != cudaSuccess)) {
    auto err_string =
        string::Sprintf("Cannot allocate %d on GPU %d, cuda status %d, %s",
                        size, dev_id, status, cudaGetErrorString(status));
    VLOG(10) << err_string;
  }

  cudaMemsetAsync(ptr, 0, size, context.stream());
  if (!TestSetValue(context)) {
    VLOG(10) << "cudamemsetaync at TestStream:" << context.stream() << " error"
             << ", ptr:" << ptr << ", dev_id:" << dev_id;
    exit(0);
  } else {
    VLOG(10) << "cudamemsetaync at TestStream:" << context.stream() << " ok"
             << ", ptr:" << ptr << ", dev_id:" << dev_id;
  }
  //}
}

inline void TestSetConstant(
    const std::vector<const framework::Scope *> &local_scopes,
    const std::string message = "") {
  VLOG(10) << "in threadedssa local_scopes size:" << local_scopes.size()
           << ", message:" << message;
  for (unsigned int i = 0; i < local_scopes.size(); i++) {
    // VLOG(10) << "in threadedssa 1";
    // auto scope = local_scopes[i];
    // framework::Scope *local_scope =
    // scope->FindVar(kLocalExecScopeName)->Get<framework::Scope *>();
    auto local_scope = local_scopes[i];
    auto var = local_scope->FindVar("embedding_para@GRAD");
    if (var == nullptr) {
      VLOG(10) << "in threadedssa can't find var:"
               << ", message:" << message;
      continue;
    }

    // VLOG(10) << "in threadedssa 2";
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();

    // VLOG(10) << "in threadedssa 2.1";
    auto slr = var->GetMutable<framework::SelectedRows>();

    // VLOG(10) << "in threadedssa 2.2";
    if (!slr->value().IsInitialized()) {
      VLOG(10) << "in threadedssa not IsInitialized:"
               << ", message:" << message;
      continue;
    }

    // VLOG(10) << "in threadedssa 3";
    auto ctx = dynamic_cast<platform::CUDADeviceContext *>(
        pool.Get(slr->value().place()));

    // VLOG(10) << "in threadedssa 3.1";
    if (ctx == nullptr) {
      VLOG(10) << "in threadedssa not cuda:"
               << ", message:" << message;
      continue;
    }

    // VLOG(10) << "in threadedssa 4";
    auto *data = slr->mutable_value()->data<float>();
    VLOG(10) << "in threadedssa prepare cudamemsetasync set:"
             << slr->value().place() << ", idx:" << i << ", data:" << data
             << ", numel:" << slr->value().numel() << ", message:" << message;

    // cudaMemsetAsync(data, 0, slr->value().numel(), ctx->stream());
    operators::math::SetConstant<platform::CUDADeviceContext, float>
        constant_functor;
    constant_functor(*ctx, slr->mutable_value(), static_cast<float>(0));
    VLOG(10) << "in threadedssa before cudamemsetasync set:"
             << slr->value().place() << ", idx:" << i
             << ", message:" << message;
    ctx->Wait();
    VLOG(10) << "in threadedssa after cudaMemsetAsync set:"
             << slr->value().place() << ", idx:" << i
             << ", message:" << message;
  }
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
