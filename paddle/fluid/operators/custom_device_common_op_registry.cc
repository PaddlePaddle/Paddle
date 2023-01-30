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
<<<<<<< HEAD
#include "paddle/fluid/operators/load_combine_op.h"
#include "paddle/fluid/operators/run_program_op.h"
#include "paddle/fluid/operators/save_combine_op.h"
#include "paddle/phi/backends/device_manager.h"
#include "paddle/phi/core/kernel_registry.h"
=======
#include "paddle/fluid/operators/run_program_op.h"
#include "paddle/fluid/operators/save_combine_op.h"
#include "paddle/phi/backends/device_manager.h"
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

#define REGISTER_OP_CUSTOM_DEVICE_KERNEL(op_type, dev_type, ...)             \
  static paddle::framework::OpKernelRegistrar<phi::CustomPlace, __VA_ARGS__> \
      __op_custom_device_kernel_registrar_##op_type##_##__acosf##__(         \
          #op_type,                                                          \
          dev_type,                                                          \
          paddle::framework::OpKernelType::kDefaultCustomizedTypeValue);     \
  __op_custom_device_kernel_registrar_##op_type##_##__acosf##__.Touch();

<<<<<<< HEAD
#define REGISTER_CUSTOM_DEVICE_GENERAL_KERNEL(                             \
    kernel_name, dev_type, layout, kernel_fn)                              \
  static phi::KernelRegistrar                                              \
      __reg_custom_device_phi_kernel_##kernel_name##_##backend##_##layout( \
          phi::RegType::INNER,                                             \
          #kernel_name,                                                    \
          dev_type,                                                        \
          DATALAYOUT(layout),                                              \
          ::phi::KernelArgsParseFunctor<decltype(&kernel_fn)>::Parse,      \
          [](const phi::KernelKey& kernel_key, phi::Kernel* kernel) {},    \
          PHI_KERNEL(kernel_fn),                                           \
          PHI_VARIADIC_KERNEL(kernel_fn))

namespace paddle {
namespace operators {

template <typename Context>
void FeedDenseTensorKernel(const Context& dev_ctx,
                           const phi::ExtendedTensor& x,
                           int col,
                           phi::DenseTensor* out);

void RegisterCustomDeviceCommonKernel(const std::string& dev_type) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
=======
namespace paddle {
namespace operators {

void RegisterCustomDeviceCommonKernel(const std::string& dev_type) {
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  auto device_type = dev_type.c_str();
  /* see [Why use single type kernel] */
  REGISTER_OP_CUSTOM_DEVICE_KERNEL(
      run_program,
      device_type,
      paddle::operators::
          RunProgramOpKernel<paddle::platform::CustomDeviceContext, float>);
  REGISTER_OP_CUSTOM_DEVICE_KERNEL(
      run_program_grad,
      device_type,
      paddle::operators ::
          RunProgramGradOpKernel<paddle::platform::CustomDeviceContext, float>);
  REGISTER_OP_CUSTOM_DEVICE_KERNEL(
      save_combine,
      device_type,
      paddle::operators ::
          SaveCombineOpKernel<paddle::platform::CustomDeviceContext, float>,
      paddle::operators ::
          SaveCombineOpKernel<paddle::platform::CustomDeviceContext, double>,
      paddle::operators ::
          SaveCombineOpKernel<paddle::platform::CustomDeviceContext, int>,
      paddle::operators ::
          SaveCombineOpKernel<paddle::platform::CustomDeviceContext, int64_t>);
<<<<<<< HEAD
  REGISTER_OP_CUSTOM_DEVICE_KERNEL(
      load_combine,
      device_type,
      paddle::operators::
          LoadCombineOpKernel<paddle::platform::CustomDeviceContext, float>,
      paddle::operators::
          LoadCombineOpKernel<paddle::platform::CustomDeviceContext, double>,
      paddle::operators::
          LoadCombineOpKernel<paddle::platform::CustomDeviceContext, int>,
      paddle::operators::
          LoadCombineOpKernel<paddle::platform::CustomDeviceContext, int8_t>,
      paddle::operators::
          LoadCombineOpKernel<paddle::platform::CustomDeviceContext, int64_t>);
  REGISTER_CUSTOM_DEVICE_GENERAL_KERNEL(
      feed_dense_tensor,
      device_type,
      ALL_LAYOUT,
      paddle::operators::FeedDenseTensorKernel<phi::CustomContext>);
#endif
=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
}

}  // namespace operators
}  // namespace paddle

#undef REGISTER_OP_CUSTOM_DEVICE_KERNEL
<<<<<<< HEAD
#undef REGISTER_CUSTOM_DEVICE_GENERAL_KERNEL
=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
