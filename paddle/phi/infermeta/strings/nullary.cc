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
#include "paddle/phi/infermeta/strings/nullary.h"

namespace phi::strings {

void CreateInferMeta(const IntArray& shape, MetaTensor* out) {
  const auto& out_dims = common::make_ddim(shape.GetData());
  out->set_dims(out_dims);
  out->set_dtype(DataType::PSTRING);
  out->set_layout(DataLayout::PSTRING_UNION);
}

}  // namespace phi::strings
