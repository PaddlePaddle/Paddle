/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/check_numerics_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/flags.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/check_numerics_utils.h"

PHI_DECLARE_int32(check_nan_inf_level);

namespace phi {

template <typename T, typename Context>
void CheckNumericsKernel(const Context& ctx,
                         const DenseTensor& tensor,
                         const std::string& op_type,
                         const std::string& var_name,
                         const int stack_height_limit,
                         const std::string& output_filepath) {
  std::string cpu_hint_str =
      phi::funcs::GetCpuHintString<T>(op_type, var_name, tensor.place());
  phi::funcs::CheckNumericsCpuImpl(tensor.data<T>(),
                                   tensor.numel(),
                                   cpu_hint_str,
                                   FLAGS_check_nan_inf_level,
                                   "cpu",
                                   output_filepath);
}

}  // namespace phi

PD_REGISTER_KERNEL(check_numerics,
                   CPU,
                   ALL_LAYOUT,
                   phi::CheckNumericsKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
