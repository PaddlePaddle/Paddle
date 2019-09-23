// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/controlflow/op_variant.h"
#include "paddle/fluid/operators/recurrent_op.h"
#include "paddle/fluid/platform/variant.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace operators {

using OpVariantSet = std::unordered_set<OpVariant, OpVariant::Hasher>;
using OpAndGradOpPair = std::pair<OpVariantSet, OpVariantSet>;

// Set vars to skip eager deletion on input recurrent and recurrent_grad for
// preparing safe eager deletion. Input contains all recurrent and
// recurrent_grad ops at block 0 and the function will find all recurrent and
// recurrent_grad ops across blocks.
void PrepareSafeEagerDeletionOnRecurrentOpAndRecurrentGradOp(
    const framework::ProgramDesc &program, OpAndGradOpPair *op_pair);

// Set vars to skip eager deletion on input recurrent and recurrent_grad for
// preparing safe eager deletion. The input block_id must be 0 and caller can
// input all ops in the block. The function will find all recurrent and
// recurrent_grad ops across blocks.
void PrepareSafeEagerDeletionOnRecurrentOpAndRecurrentGradOp(
    const framework::ProgramDesc &program, int block_id,
    const std::vector<std::unique_ptr<paddle::framework::OperatorBase>>
        &all_ops);

}  // namespace operators
}  // namespace paddle
