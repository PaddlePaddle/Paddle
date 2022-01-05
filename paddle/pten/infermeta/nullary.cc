/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

// See Note [ Why still include the fluid headers? ]
#include "paddle/pten/infermeta/nullary.h"

namespace pten {

DenseTensorMeta CreateInferMeta(const std::vector<int64_t>& shape,
                                DataType dtype,
                                DataLayout layout) {
  const auto& out_dims = paddle::framework::make_ddim(shape);
  return {dtype, out_dims, layout};
}

DenseTensorMeta CreateInferMeta(const ScalarArray& shape,
                                DataType dtype,
                                DataLayout layout) {
  const auto& out_dims = paddle::framework::make_ddim(shape.GetData());
  return {dtype, out_dims, layout};
}

}  // namespace pten
