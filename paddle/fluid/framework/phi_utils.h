/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/op_kernel_type.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/macros.h"
#include "paddle/fluid/platform/place.h"

#include "paddle/fluid/framework/operator.h"
#include "paddle/phi/api/lib/utils/tensor_utils.h"
#include "paddle/phi/common/backend.h"
#include "paddle/phi/core/compat/arg_map_context.h"
#include "paddle/phi/core/kernel_factory.h"
#include "paddle/utils/flat_hash_map.h"
#include "paddle/utils/small_vector.h"

#ifdef PADDLE_WITH_XPU
#include "paddle/fluid/platform/device/xpu/xpu_op_list.h"
#endif

namespace paddle {
namespace framework {

/* Kernel Key translate */

OpKernelType TransPhiKernelKeyToOpKernelType(const phi::KernelKey& kernel_key);
phi::KernelKey TransOpKernelTypeToPhiKernelKey(const OpKernelType& kernel_type);
phi::KernelKey FallBackToCpu(const OpKernelType& expected_kernel_key,
                             const phi::KernelKey& kernel_key,
                             const framework::OperatorBase& op);

/* Kernel Args parse */

class KernelArgsNameMaker {
 public:
  virtual ~KernelArgsNameMaker() {}
  virtual const paddle::SmallVector<const char*>& GetInputArgsNames() = 0;
  virtual const paddle::SmallVector<const char*>& GetOutputArgsNames() = 0;
  virtual const paddle::SmallVector<const char*>& GetAttrsArgsNames() = 0;
};

void InitDefaultKernelSignatureMap();

// TODO(Wilber): support others device context.
template <typename T>
struct ConvertToPhiContext {
  using TYPE = T;
};

template <>
struct ConvertToPhiContext<platform::CPUDeviceContext> {
  using TYPE = phi::CPUContext;
};

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
template <>
struct ConvertToPhiContext<platform::CUDADeviceContext> {
  using TYPE = phi::GPUContext;
};
#endif

#ifdef PADDLE_WITH_XPU
template <>
struct ConvertToPhiContext<platform::XPUDeviceContext> {
  using TYPE = phi::XPUContext;
};
#endif

}  // namespace framework
}  // namespace paddle
