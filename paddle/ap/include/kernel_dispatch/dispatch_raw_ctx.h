// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/ap/include/adt/adt.h"
#include "paddle/ap/include/axpr/type.h"
#include "paddle/ap/include/code_module/data_type.h"
#include "paddle/ap/include/kernel_dispatch/arg_value.h"
#include "paddle/ap/include/kernel_dispatch/device_ctx.h"
#include "paddle/ap/include/kernel_dispatch/typed_buffer.h"
#include "paddle/ap/include/rt_module/module.h"
#include "paddle/phi/core/dense_tensor.h"

namespace phi {

class DenseTensor;

}

namespace ap::kernel_dispatch {

using RtModuleImpl = std::variant<std::shared_ptr<const rt_module::Module>>;

struct RtModule : public RtModuleImpl {
  using RtModuleImpl::RtModuleImpl;
  ADT_DEFINE_VARIANT_METHODS(RtModuleImpl);
};

template <typename ValueT>
struct DispatchRawCtxImpl {
  DeviceCtx device_ctx;
  adt::List<ValueT> inputs;
  adt::List<ValueT> outputs;
  RtModule rt_module;

  bool operator==(const DispatchRawCtxImpl& other) const {
    return &other == this;
  }

  Result<adt::Ok> LaunchCudaKernel(
      const std::string& func_name,
      int64_t num_blocks,
      int64_t num_threads,
      const adt::List<ArgValue>& kernel_args) const;
};

template <typename ValueT>
ADT_DEFINE_RC(DispatchRawCtx, DispatchRawCtxImpl<ValueT>);

}  // namespace ap::kernel_dispatch

namespace ap::axpr {

template <typename ValueT>
struct TypeImpl<ap::kernel_dispatch::DispatchRawCtx<ValueT>>
    : public std::monostate {
  using value_type = ap::kernel_dispatch::DispatchRawCtx<ValueT>;

  const char* Name() const { return "DispatchRawCtx"; }
};

}  // namespace ap::axpr
