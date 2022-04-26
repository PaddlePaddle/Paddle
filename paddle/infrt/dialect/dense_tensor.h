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
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <string>

#include "paddle/infrt/dialect/infrt/ir/infrt_dialect.h"

#include "paddle/infrt/dialect/dense_tensor_dialect.hpp.inc"

#define GET_OP_CLASSES
#include "paddle/infrt/dialect/dense_tensor.hpp.inc"
