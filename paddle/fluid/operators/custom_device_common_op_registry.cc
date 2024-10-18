/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/custom_device_common_op_registry.h"
#include "paddle/fluid/distributed/collective/process_group.h"
#include "paddle/fluid/operators/collective/c_concat_op.h"
#include "paddle/fluid/operators/load_combine_op.h"
#include "paddle/fluid/operators/save_combine_op.h"
#include "paddle/phi/api/backward/backward_api.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/backends/device_manager.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/distributed/xccl_comm_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/platform/collective_helper.h"
#include "paddle/phi/kernels/funcs/axis_utils.h"

#define REGISTER_OP_CUSTOM_DEVICE_KERNEL(op_type, dev_type, ...)             \
  static paddle::framework::OpKernelRegistrar<phi::CustomPlace, __VA_ARGS__> \
      __op_custom_device_kernel_registrar_##op_type##_##__acosf##__(         \
          #op_type,                                                          \
          dev_type,                                                          \
          paddle::framework::OpKernelType::kDefaultCustomizedTypeValue);     \
  __op_custom_device_kernel_registrar_##op_type##_##__acosf##__.Touch();

#define REGISTER_CUSTOM_DEVICE_GENERAL_KERNEL(                             \
    kernel_name, dev_type, layout, kernel_fn)                              \
  static phi::KernelRegistrar                                              \
      __reg_custom_device_phi_kernel_##kernel_name##_##backend##_##layout( \
          phi::RegType::INNER,                                             \
          #kernel_name,                                                    \
          dev_type,                                                        \
          DATA_LAYOUT(layout),                                             \
          ::phi::KernelArgsParseFunctor<decltype(&kernel_fn)>::Parse,      \
          [](const phi::KernelKey& kernel_key, phi::Kernel* kernel) {},    \
          PHI_KERNEL(kernel_fn),                                           \
          PHI_VARIADIC_KERNEL(kernel_fn))

namespace paddle {
namespace operators {

void RegisterCustomDeviceCommonKernel(const std::string& dev_type) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  auto device_type = dev_type.c_str();
  /* see [Why use single type kernel] */
  REGISTER_OP_CUSTOM_DEVICE_KERNEL(
      save_combine,
      device_type,
      paddle::operators ::SaveCombineOpKernel<phi::CustomContext, float>,
      paddle::operators ::SaveCombineOpKernel<phi::CustomContext, double>,
      paddle::operators ::SaveCombineOpKernel<phi::CustomContext, int>,
      paddle::operators ::SaveCombineOpKernel<phi::CustomContext, int64_t>);
  REGISTER_OP_CUSTOM_DEVICE_KERNEL(
      load_combine,
      device_type,
      paddle::operators::LoadCombineOpKernel<float, phi::CustomContext>,
      paddle::operators::LoadCombineOpKernel<double, phi::CustomContext>,
      paddle::operators::LoadCombineOpKernel<int, phi::CustomContext>,
      paddle::operators::LoadCombineOpKernel<int8_t, phi::CustomContext>,
      paddle::operators::LoadCombineOpKernel<int64_t, phi::CustomContext>);
#endif
}

}  // namespace operators
}  // namespace paddle

#undef REGISTER_OP_CUSTOM_DEVICE_KERNEL
#undef REGISTER_CUSTOM_DEVICE_GENERAL_KERNEL
