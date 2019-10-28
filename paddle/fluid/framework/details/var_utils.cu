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

#include "paddle/fluid/framework/details/var_utils.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace framework {
namespace details {

template <typename T>
struct NanInfRangeFunctor {
  explicit NanInfRangeFunctor(const T* a) : a_(a) {}
  inline HOSTDEVICE void operator()(size_t id) const {
    float r = static_cast<float>(a_[id]);
    if (std::isnan(r) || std::isinf(r)) {
      printf("idx:%u value:%f nan or inf\n", id, r);
      PADDLE_ENFORCE(0, "find nan or inf");
    }
    // PADDLE_ENFORCE(std::isnan(r) == 0, "%f is nan", r);
    // PADDLE_ENFORCE(std::isinf(r) == 0, "%f is inf", r);
  }
  const T* a_;
};

struct TensorCheckerVisitor {
  TensorCheckerVisitor(const std::string& var_name,
                       const framework::Tensor& tensor,
                       const platform::Place& place)
      : var_name_(var_name), tensor_(tensor), place_(place) {}

  template <typename T>
  void apply() const {
    if (!std::is_floating_point<T>::value) {
      VLOG(10) << var_name_
               << " need not to check, it's type is not float point";
      return;
    }

    NanInfRangeFunctor<T> func(tensor_.data<T>());
    if (platform::is_gpu_place(tensor_.place())) {
#ifdef PADDLE_WITH_CUDA
      auto* dev_ctx = reinterpret_cast<platform::CUDADeviceContext*>(
          platform::DeviceContextPool::Instance().Get(tensor_.place()));
      platform::ForRange<platform::CUDADeviceContext> for_range(
          *dev_ctx, tensor_.numel());
      for_range(func);
#else
      PADDLE_THROW("PaddlePaddle should compile with GPU.");
#endif
      return;
    }

    auto* dev_ctx = reinterpret_cast<platform::CPUDeviceContext*>(
        platform::DeviceContextPool::Instance().Get(tensor_.place()));
    platform::ForRange<platform::CPUDeviceContext> for_range(*dev_ctx,
                                                             tensor_.numel());
    for_range(func);
  }

  std::string var_name_;
  const framework::Tensor& tensor_;
  const platform::Place& place_;
};

void EnforceNoNanOrInf(const std::string& op_type,
                       const framework::Scope& scope,
                       const std::string& var_name,
                       const platform::Place& place) {
  auto* var = scope.FindVar(var_name);
  PADDLE_ENFORCE_NOT_NULL(var, "can't find var:%s", var_name);

  const Tensor* tensor{nullptr};
  if (var->IsType<framework::LoDTensor>()) {
    tensor = &var->Get<framework::LoDTensor>();
  } else if (var->IsType<framework::SelectedRows>()) {
    tensor = &var->Get<framework::SelectedRows>().value();
  } else {
    VLOG(10) << var_name << " var_name need not to check";
    return;
  }

  if (tensor->memory_size() == 0) {
    VLOG(10) << var_name << " var_name need not to check, size == 0";
    return;
  }

  VLOG(10) << "begin check " << op_type << " var_name:" << var_name
           << ", place:" << tensor->place << ", numel:" << tensor->numel();
  TensorCheckerVisitor vistor(var_name, *tensor, place);
  VisitDataType(tensor->type(), vistor);
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
