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

#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/include/infermeta.h"
#include "paddle/pten/kernels/cpu/creation.h"
#include "paddle/pten/kernels/cuda/creation.h"

namespace pten {

// TODO(YuanRisheng) This function name should be same as User API name.
// TODO(zyfncg) Automatic code generation
template <typename T, typename ContextT>
DenseTensor FillAnyLike(
    const ContextT& dev_ctx,
    const DenseTensor& x,
    const Scalar& val,
    DataType dtype = DataType::UNDEFINED,
    Backend backend = Backend::UNDEFINED,  // Is backend needed here?
    DataLayout layout = DataLayout::UNDEFINED) {
  auto out_meta = FullLikeInferMeta(x.meta(), dtype, layout);
  const auto allocator =
      std::make_shared<paddle::experimental::DefaultAllocator>(
          dev_ctx.GetPlace());
  pten::DenseTensor dense_out(allocator, out_meta);
  FillAnyLike<T>(dev_ctx, val, &dense_out);
  return dense_out;
}

}  // namespace pten
