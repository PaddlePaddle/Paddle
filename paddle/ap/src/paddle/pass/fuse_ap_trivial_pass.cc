// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/ap/include/paddle/pass/fuse_ap_trivial_pass.h"
#include "paddle/ap/include/paddle/pass/move_trivial_fusion_range_to_fusion_op_pass.h"

namespace ap::paddle {

std::unique_ptr<::pir::Pass> CreateFuseApTrivialPass() {
  return CreateMoveTrivialFusionRangeToFusionOpPass();
}

}  // namespace ap::paddle
