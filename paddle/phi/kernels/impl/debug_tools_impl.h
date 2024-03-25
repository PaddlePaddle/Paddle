// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include "glog/logging.h"
#include "paddle/common/flags.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/funcs/math_function.h"

COMMON_DECLARE_bool(check_nan_inf);

namespace phi {

template <typename T, typename Context>
void CheckModelNanInfKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            int flag,
                            DenseTensor* out) {
  phi::CastKernel<T>(dev_ctx, x, x.dtype(), out);
  VLOG(6) << "model_check_nan_inf: Change FLAGS_check_nan_inf "
          << FLAGS_check_nan_inf << " to " << flag;
  FLAGS_check_nan_inf = flag;
}

}  // namespace phi
