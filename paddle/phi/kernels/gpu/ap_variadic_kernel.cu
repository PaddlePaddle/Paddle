// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include <mutex>
#include <unordered_map>
#include "glog/logging.h"
#include "jitify.hpp"  // NOLINT
#include "paddle/common/enforce.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_device_function.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/impl/activation_grad_impl.h"
#include "paddle/phi/kernels/impl/activation_impl.h"

#include "paddle/ap/include/axpr/data_type_util.h"
#include "paddle/ap/include/kernel_dispatch/ap_variadic_kernel.h"
#include "paddle/ap/include/paddle/phi/device_ctx.h"

namespace phi {

template <typename Context>
void AllocateOutTensors(const Context& dev_ctx,
                        const std::vector<DenseTensor*>& outs) {
  for (auto* out : outs) {
    auto out_dtype = ap::axpr::GetDataTypeFromPhiDataType(out->dtype());
    PADDLE_ENFORCE_EQ(
        out_dtype.HasOkValue(),
        true,
        phi::errors::InvalidArgument("GetDataTypeFromPhiDataType() failed !"));

    out_dtype.GetOkValue().Match(
        [&](const ap::axpr::CppDataType<ap::adt::Undefined>&) {
          PADDLE_THROW(
              phi::errors::InvalidArgument("allocate not support undefined !"));
        },
        [&](const ap::axpr::CppDataType<uint64_t>&) {
          PADDLE_THROW(
              phi::errors::InvalidArgument("allocate not support uint64_t !"));
        },
        [&](const ap::axpr::CppDataType<uint32_t>&) {
          PADDLE_THROW(
              phi::errors::InvalidArgument("allocate not support uint32_t !"));
        },
        [&](const ap::axpr::CppDataType<uint16_t>&) {
          PADDLE_THROW(
              phi::errors::InvalidArgument("allocate not support uint16_t !"));
        },
        [&](const auto& impl) {
          using tensor_type = typename std::decay_t<decltype(impl)>::type;
          dev_ctx.template Alloc<tensor_type>(out);
        });
  }
  return;
}

template <typename T, typename Context>
void ApVariadicKernel(const Context& dev_ctx,
                      const std::vector<const DenseTensor*>& xs,
                      int num_outputs,
                      const std::string& code_module_lambda,
                      const std::string& infer_meta_lambda,
                      const std::string& kernel_dispatch_lambda,
                      const std::string& kernel_dispatch_const_data_lambda,
                      std::vector<DenseTensor*> outs) {
  PADDLE_ENFORCE_GT(
      xs.size(),
      0,
      phi::errors::InvalidArgument(
          "At least 1 input is required. current number out uts: // %d",
          xs.size()));
  PADDLE_ENFORCE_GT(
      outs.size(),
      0,
      phi::errors::InvalidArgument(
          "num_outputs must be greater than 1. current _outputs: // %d",
          outs.size()));

  AllocateOutTensors<Context>(dev_ctx, outs);
  std::shared_ptr<ap::kernel_dispatch::DeviceCtxImpl> impl =
      std::make_shared<ap::paddle::DeviceCtx<Context>>(&dev_ctx);
  ap::kernel_dispatch::DeviceCtx ap_device_ctx{impl};
  const auto& ret =
      ap::kernel_dispatch::ApVariadicKernel(ap_device_ctx,
                                            xs,
                                            num_outputs,
                                            code_module_lambda,
                                            infer_meta_lambda,
                                            kernel_dispatch_lambda,
                                            kernel_dispatch_const_data_lambda,
                                            outs);
  PADDLE_ENFORCE(
      !ret.HasError(),
      "ap_kernel failed. \nTraceback (most recent call last):\n%s\n%s: %s. ",
      ret.GetError().CallStackToString(),
      ret.GetError().class_name(),
      ret.GetError().msg());
}

}  // namespace phi

#ifdef PADDLE_WITH_HIP
PD_REGISTER_KERNEL(ap_variadic,
                   GPU,
                   ALL_LAYOUT,
                   phi::ApVariadicKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
#else
PD_REGISTER_KERNEL(ap_variadic,
                   GPU,
                   ALL_LAYOUT,
                   phi::ApVariadicKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#endif
