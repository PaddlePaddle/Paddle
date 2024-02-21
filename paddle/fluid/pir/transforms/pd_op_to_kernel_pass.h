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

#include "paddle/phi/common/place.h"
#include "paddle/pir/include/core/program.h"

namespace paddle {
namespace dialect {

pir::Type ConvertOpTypeToKernelType(pir::Type op_type);

TEST_API std::unique_ptr<pir::Program> PdOpLowerToKernelPass(
    pir::Program* prog, phi::Place place = phi::CPUPlace());

void ProcessBlock(
    const phi::Place& place,
    pir::Block* block,
    pir::Block* new_block,
    pir::IrContext* ctx,
    std::unordered_map<pir::Operation*, pir::Operation*>* map_op_pair,
    std::unordered_map<pir::Value, pir::Value>* map_value_pair,
    bool for_if_block = false);
}  // namespace dialect
}  // namespace paddle
