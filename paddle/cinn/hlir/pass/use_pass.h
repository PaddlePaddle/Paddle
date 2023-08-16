// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/common/macros.h"
#include "paddle/cinn/hlir/pass/use_general_pass.h"

CINN_USE_REGISTER(InferShape)
CINN_USE_REGISTER(OpFusion)
CINN_USE_REGISTER(AlterLayout)
CINN_USE_REGISTER(ConstPropagate)

CINN_USE_REGISTER(DCE)
CINN_USE_REGISTER(DotMerger)
CINN_USE_REGISTER(OpFusionPass)
CINN_USE_REGISTER(FusionMergePass)
CINN_USE_REGISTER(GeneralFusionMergePass)
CINN_USE_REGISTER(CheckFusionAccuracyPass)

CINN_USE_REGISTER(CommonSubexpressionEliminationPass)
CINN_USE_REGISTER(TransToCustomCallPass)
CINN_USE_REGISTER(DenseMergePass)
CINN_USE_REGISTER(ConstantFolding)
CINN_USE_REGISTER(ReduceSplit)
CINN_USE_REGISTER(SingleGroupOptimizePass)
