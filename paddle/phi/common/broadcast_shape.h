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

#include <string>
#include <typeindex>
#include <typeinfo>
#include <utility>

#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/attribute.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/macros.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/core/type_defs.h"
#include "paddle/utils/any.h"
#include "paddle/utils/flat_hash_map.h"
#include "paddle/utils/small_vector.h"

namespace phi {

namespace {

template <typename Container>
Container InferShapeImpl(DDim a, DDim b) {
  int DimsA{a.size()};
  int DimsB{b.size()};
  int ndim{DimsA > DimsB ? DimsA : DimsB};
  Container expandedSizes(ndim);

  for (ptrdiff_t i{ndim - 1}; i >= 0; --i) {
    ptrdiff_t offset{ndim - 1 - i};
    ptrdiff_t dimA{DimsA - 1 - offset};
    ptrdiff_t dimB{DimsB - 1 - offset};

    int64_t sizeA{dimA >= 0 ? a[dimA] : 1};
    int64_t sizeB{dimB >= 0 ? b[dimB] : 1};

    PADDLE_ENFORCE_EQ(
        true,
        sizeA == sizeB || sizeA == 1 || sizeB == 1,
        phi::errors::InvalidArgument("The size of tensor a (",
                                     sizeA,
                                     ") must match the size of tensor b (",
                                     sizeB,
                                     ") at non-singleton dimension ",
                                     i));

    // 1s map to the other size (even 0).
    expandedSizes[i] = sizeA == 1 ? sizeB : sizeA;
  }

  return expandedSizes;
}

}  // namespace

std::vector<int64_t> InferBroadcastShape(DDim a, DDim b) {
  return InferShapeImpl<std::vector<int64_t>>(a, b);
}

}  // namespace phi
