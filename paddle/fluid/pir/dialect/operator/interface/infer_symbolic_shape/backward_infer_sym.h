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

#include "paddle/pir/include/dialect/shape/utils/shape_analysis.h"

namespace paddle::dialect {
OP_DECLARE_INFER_SYMBOLIC_SHAPE(FusedAttentionGrad)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(GroupNormGrad)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(GroupNormGrad_)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Conv2dGrad)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(MatmulGrad)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(DepthwiseConv2dGrad)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Pool2dGrad)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(BceLossGrad)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(BceLossGrad_)

}  // namespace paddle::dialect
