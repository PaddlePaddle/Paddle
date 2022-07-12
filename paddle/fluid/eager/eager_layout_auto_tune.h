// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"
#include "paddle/fluid/eager/eager_layout_transformer.h"
#include "paddle/fluid/imperative/layout_autotune.h"
namespace egr {

inline LayoutTransformer EagerAutotuneLayoutTransformer(
    const std::string op_name,
    const paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               kSlotSmallVectorSize>& amp_tensors_vector) {
  VLOG(4) << "Layout asdf AmpAutoCasts: inputs(" << op_name;
  auto transposer = LayoutTransformer(op_name);
  return transposer;
}

template <typename T>  // default int
inline LayoutTransformer EagerAutotuneLayoutTransformer(
    const std::string& op_name,
    const paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               kSlotSmallVectorSize>& amp_tensors_vector,
    T attrs) {
  VLOG(4) << "EagerAutotuneLayoutTransformer op_name : " << op_name;
  auto transposer = LayoutTransformer(op_name);
  transposer.SetAttr<T>(attrs);
  return transposer;
}

template <typename T1, typename T2>  // default int
inline LayoutTransformer EagerAutotuneLayoutTransformer(
    const std::string& op_name,
    const paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               kSlotSmallVectorSize>& amp_tensors_vector,
    T1 attr1,
    T2 attr2) {
  VLOG(4) << "Layout asdf AmpAutoCasts: inputs(" << op_name;
  auto transposer = LayoutTransformer(op_name);
  transposer.SetAttr<T1, T2>(attr1, attr2);
  return transposer;
}

}  // namespace egr
