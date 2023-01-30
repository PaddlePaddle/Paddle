// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/abs_grad_kernel.h"

#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/float16.h"
<<<<<<< HEAD
#include "paddle/phi/common/type_traits.h"
=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/abs_grad_kernel_impl.h"

using phi::dtype::complex;

PD_REGISTER_KERNEL(abs_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::AbsGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
<<<<<<< HEAD
                   phi::dtype::bfloat16,
                   complex<float>,
                   complex<double>) {
  kernel->InputAt(1).SetDataType(phi::dtype::ToReal(kernel_key.dtype()));
}
=======
                   complex<float>,
                   complex<double>) {}
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
PD_REGISTER_KERNEL(abs_double_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::AbsDoubleGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   complex<float>,
<<<<<<< HEAD
                   complex<double>) {
  kernel->InputAt(1).SetDataType(phi::dtype::ToReal(kernel_key.dtype()));
}
=======
                   complex<double>) {}
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
