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

#include "paddle/fluid/inference/tensorrt/dynamic_shape_infermeta_factory.h"

namespace paddle {
namespace inference {
namespace tensorrt {
DynamicMetaFnFactory& DynamicMetaFnFactory::Instance() {
  static DynamicMetaFnFactory g_meta_fn_map;
  return g_meta_fn_map;
}

void DynamicMetaFnFactory::Insert(std::string op_name,
                                  DynamicMetaFn infer_meta_fn) {
  PADDLE_ENFORCE_NE(
      Contains(op_name),
      true,
      phi::errors::AlreadyExists(
          "`%s` op's DynamicInferMetaFn has been registered.", op_name));
  meta_fn_map_.insert({std::move(op_name), std::move(infer_meta_fn)});
}

const DynamicMetaFn& DynamicMetaFnFactory::Get(
    const std::string& op_name) const {
  auto it = meta_fn_map_.find(op_name);
  PADDLE_ENFORCE_NE(
      it,
      meta_fn_map_.end(),
      phi::errors::NotFound("`%s` op's DynamicInferMetaFn has been registered.",
                            op_name));
  return it->second;
}
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
