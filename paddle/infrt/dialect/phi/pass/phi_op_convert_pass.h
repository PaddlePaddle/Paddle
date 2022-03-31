// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include <mlir/Pass/Pass.h>
#include "paddle/infrt/dialect/infrt/common/types.h"

namespace infrt {
/*
 * phiOpCvtPass.
 * Convert the general operators from pd Dialect to phi dialect.
 */
std::unique_ptr<mlir::Pass> CreatePhiOpCvtPass(std::vector<Place> valid_places);

std::unique_ptr<mlir::Pass> CreatePhiOpCvtPass();

}  // namespace infrt
