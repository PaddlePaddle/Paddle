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

CINN_USE_REGISTER(relu_decomposers)
CINN_USE_REGISTER(relu_grad_decomposers)
CINN_USE_REGISTER(gelu_decomposers)
CINN_USE_REGISTER(softmax_decomposers)
CINN_USE_REGISTER(sum_decomposers)
CINN_USE_REGISTER(broadcast_decomposers)
CINN_USE_REGISTER(broadcast_grad_decomposers)
CINN_USE_REGISTER(batch_norm_decomposer)
CINN_USE_REGISTER(batch_norm_train_decomposer)
CINN_USE_REGISTER(batch_norm_grad_decomposer)
CINN_USE_REGISTER(top_k_decomposer)
