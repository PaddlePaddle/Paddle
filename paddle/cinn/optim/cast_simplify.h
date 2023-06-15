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

#include "cinn/ir/ir.h"

namespace cinn::optim {

/**
 * Simplify the Cast nodes.
 *
 * There are several patterns:
 * 1. the source and target type are the same, drop the Cast node
 * 2. for intermediate numbers, just replace the Cast node with a Node of the target type
 */
void CastSimplify(Expr* e);

}  // namespace cinn::optim
