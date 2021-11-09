// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/pten/kernels/cpu/cast.h"
#include "paddle/pten/common/data_type.h"
#include "paddle/pten/core/kernel_registry.h"

#include "paddle/fluid/platform/transform.h"

namespace pten {

namespace detail {

template <typename InT, typename OutT>
void cast_cpu_kernel(const CPUContext& dev_ctx,
                     const DenseTensor& x,
                     DenseTensor* out) {
  auto* in_begin = x.data<InT>();
  auto numel = x.numel();
  auto* in_end = in_begin + numel;

  auto* out_begin = out->mutable_data<OutT>();

  paddle::platform::Transform<CPUContext> trans;
  trans(dev_ctx,
        in_begin,
        in_end,
        out_begin,
        CastOpTransformFunctor<InT, OutT>());
}

}  // namespace detail

}  // namespace pten

PT_REGISTER_MODULE(CastCPU);
