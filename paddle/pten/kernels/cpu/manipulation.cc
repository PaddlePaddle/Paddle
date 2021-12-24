//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/pten/kernels/cpu/manipulation.h"
#include "paddle/pten/api/ext/dispatch.h"
#include "paddle/pten/infermeta/unary.h"
#include "paddle/pten/kernels/cpu/utils.h"
#include "paddle/pten/kernels/hybird/math/cast_func.h"

namespace pten {

template <typename T>
void Cast(const CPUContext& dev_ctx,
          const DenseTensor& x,
          DataType out_dtype,
          DataType in_dtype,
          DenseTensor* out) {
  PD_VISIT_ALL_TYPES(out_dtype, "CastKernelImpl", ([&] {
                       math::CastKernelImpl<CPUContext, T, data_t>(
                           dev_ctx, x, out);
                     }));
}

}  // namespace pten

PT_REGISTER_KERNEL(cast,
                   CPU,
                   ALL_LAYOUT,
                   pten::Cast,
                   float,
                   double,
                   int,
                   int64_t,
                   int16_t,
                   bool,
                   uint8_t,
                   paddle::platform::float16,
                   paddle::platform::bfloat16,
                   paddle::platform::complex<float>,
                   paddle::platform::complex<double>) {
  kernel->OutputAt(0).SetDataType(paddle::experimental::DataType::UNDEFINED);
}
