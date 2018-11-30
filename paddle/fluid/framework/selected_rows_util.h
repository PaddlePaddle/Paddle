/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace framework {

template <typename DataType>
void SelectedRowsCopy(const SelectedRows& src, const platform::Place& dst_place,
                      SelectedRows* dst) {
  auto& out = *dst;
  out.set_rows(src.rows());
  out.set_height(src.height());

  auto dims = framework::make_ddim(
      {static_cast<int64_t>(src.rows().size()), src.value().dims()[1]});

  out.mutable_value()->mutable_data<DataType>(dims, dst_place);

  TensorCopy(src.value(), dst_place, dst->mutable_value());
}

};  // namespace framework
};  // namespace paddle
