// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/eager/api/all.h"
#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"
#include "paddle/fluid/eager/api/manual/eager_manual/dygraph_forward_api.h"
#include "paddle/fluid/primitive/backend/generated/generated_backend.h"
#include "paddle/fluid/primitive/backend/manual/manual_prim_backend.h"

namespace paddle::primitive::backend {

template <>
Tensor full<Tensor>(const IntArray& shape,
                    const Scalar& value,
                    DataType dtype,
                    Place place) {
  VLOG(4) << "Eager Prim API full_ad_func call";
  if (place.GetType() == AllocationType::UNDEFINED) {
    return ::full_ad_func(shape, value, dtype, CPUPlace());
  } else {
    return ::full_ad_func(shape, value, dtype, place);
  }
}

template <>
Tensor arange<Tensor>(const Tensor& start,
                      const Tensor& end,
                      const Tensor& step,
                      DataType dtype,
                      Place place) {
  VLOG(4) << "Eager Prim API arange_ad_func call";
  return ::arange_ad_func(start, end, step, dtype, place);
}

}  // namespace paddle::primitive::backend
