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

#pragma once

// See Note: [ How do we organize the kernel directory ]
#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/include/infershape.h"
#include "paddle/pten/kernels/cpu/nn.h"
#include "paddle/pten/kernels/cuda/nn.h"

namespace pten {

template <typename T, typename ContextT>
DenseTensor ElementwiseAdd(const ContextT& dev_ctx,
                           const DenseTensor& x,
                           const DenseTensor& y,
                           int axis) {
  auto out_meta = ElementwiseInferShape(x.meta(), y.meta(), axis);
  const auto allocator =
      std::make_shared<paddle::experimental::DefaultAllocator>(
          dev_ctx.GetPlace());
  pten::DenseTensor dense_out(allocator, out_meta);
  ElementwiseAdd<T>(dev_ctx, x, y, axis, &dense_out);
  return dense_out;
}

}  // namespace pten
