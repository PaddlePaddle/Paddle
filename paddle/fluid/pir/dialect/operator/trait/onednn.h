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

#ifdef PADDLE_WITH_DNNL

#include "paddle/pir/include/core/op_base.h"

namespace paddle {
namespace dialect {
class OneDNNTrait : public pir::OpTraitBase<OneDNNTrait> {
 public:
  explicit OneDNNTrait(const pir::Operation *op)
      : pir::OpTraitBase<OneDNNTrait>(op) {}
};

class OneDNNOnlyTrait : public pir::OpTraitBase<OneDNNOnlyTrait> {
 public:
  explicit OneDNNOnlyTrait(const pir::Operation *op)
      : pir::OpTraitBase<OneDNNOnlyTrait>(op) {}
};

class OneDNNDynamicFallbackTrait
    : public pir::OpTraitBase<OneDNNDynamicFallbackTrait> {
 public:
  explicit OneDNNDynamicFallbackTrait(const pir::Operation *op)
      : pir::OpTraitBase<OneDNNDynamicFallbackTrait>(op) {}
};

}  // namespace dialect
}  // namespace paddle

IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::OneDNNTrait)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::OneDNNOnlyTrait)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::OneDNNDynamicFallbackTrait)

#endif
