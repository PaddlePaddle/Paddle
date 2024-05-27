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

#pragma once

#include <memory>

#include "paddle/pir/include/pass/pass.h"

namespace pir {
class Value;
class PatternRewriter;
}  // namespace pir
namespace cinn {
namespace dialect {
class GenerateShapeOp;

namespace ir {

namespace details {
std::optional<pir::Value> GetOutReplacement(cinn::dialect::GenerateShapeOp op,
                                            pir::PatternRewriter* rewriter);
}  // namespace details

std::unique_ptr<pir::Pass> CreateSplitGenerateShapeIntoShapeOpsPass();

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
