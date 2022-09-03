//   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/cpu/cpu_context.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/platform/transform.h"

namespace phi {

template <typename InT, typename OutT>
struct CastOpTransformFunctor {
  HOSTDEVICE OutT operator()(InT in) const { return static_cast<OutT>(in); }
};

template <typename InT, typename OutT>
void CastKernelImpl(const CPUContext& dev_ctx,
                    const DenseTensor& x,
                    DenseTensor* out) {
  auto* in_begin = x.data<InT>();
  auto numel = x.numel();
  auto* in_end = in_begin + numel;

  auto* out_begin = dev_ctx.Alloc<OutT>(out);

  paddle::platform::Transform<CPUContext> trans;
  trans(dev_ctx,
        in_begin,
        in_end,
        out_begin,
        CastOpTransformFunctor<InT, OutT>());
}

}  // namespace phi
