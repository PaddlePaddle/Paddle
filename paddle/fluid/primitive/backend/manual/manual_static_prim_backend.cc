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

#include "paddle/fluid/pir/dialect/operator/ir/manual_api.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_api.h"
#include "paddle/fluid/primitive/backend/generated/generated_backend.h"
#include "paddle/fluid/primitive/backend/manual/manual_prim_backend.h"
#include "paddle/fluid/primitive/primitive/generated_primitive.h"
#include "paddle/fluid/primitive/primitive/manual_primitive.h"
#include "paddle/fluid/primitive/type/lazy_tensor.h"

namespace paddle {
namespace primitive {
namespace backend {

template <>
Tensor full<LazyTensor>(const IntArray& shape,
                        const Scalar& value,
                        DataType dtype,
                        Place place) {
  auto op_res =
      paddle::dialect::full(shape.GetData(), value.to<float>(), dtype, place);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out;
}

}  // namespace backend
}  // namespace primitive
}  // namespace paddle
