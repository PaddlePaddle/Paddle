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
#include "paddle/pten/api/lib/utils/tensor_utils.h"
#include "paddle/pten/common/backend.h"
#include "paddle/pten/core/compat/arg_map_context.h"
#include "paddle/pten/core/kernel_factory.h"
#include "paddle/utils/flat_hash_map.h"
#include "paddle/utils/small_vector.h"

#ifdef PADDLE_WITH_XPU
#include "paddle/fluid/platform/device/xpu/xpu_op_list.h"
#endif

namespace paddle {
namespace framework {

using KernelSignature = pten::KernelSignature;

/* Kernel Key translate */

OpKernelType TransPtenKernelKeyToOpKernelType(
    const pten::KernelKey& kernel_key);
pten::KernelKey TransOpKernelTypeToPtenKernelKey(
    const OpKernelType& kernel_type);
pten::KernelKey FallBackToCpu(const OpKernelType& expected_kernel_key,
                              const pten::KernelKey& kernel_key,
                              const framework::OperatorBase& op);

/* Kernel Args parse */

class KernelArgsNameMaker {
 public:
  virtual ~KernelArgsNameMaker() {}
  virtual const paddle::SmallVector<std::string>& GetInputArgsNames() = 0;
  virtual const paddle::SmallVector<std::string>& GetOutputArgsNames() = 0;
  virtual const paddle::SmallVector<std::string>& GetAttrsArgsNames() = 0;
};

void InitDefaultKernelSignatureMap();

void SetAllocationForOutputTenosr(pten::TensorBase* tensor,
                                  const platform::Place& place);

// TODO(Wilber): support others device context.
template <typename T>
struct ConvertToPtenContext {
  using TYPE = T;
};

template <>
struct ConvertToPtenContext<platform::CPUDeviceContext> {
  using TYPE = pten::CPUContext;
};

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
template <>
struct ConvertToPtenContext<platform::CUDADeviceContext> {
  using TYPE = pten::GPUContext;
};
#endif

#ifdef PADDLE_WITH_XPU
template <>
struct ConvertToPtenContext<platform::XPUDeviceContext> {
  using TYPE = pten::XPUContext;
};
#endif

}  // namespace framework
}  // namespace paddle
