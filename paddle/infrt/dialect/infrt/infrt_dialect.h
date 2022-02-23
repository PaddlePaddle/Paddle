// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//
#include <llvm/ADT/StringMap.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include "paddle/infrt/dialect/infrt/common_type.h"

#include "paddle/infrt/dialect/infrt/infrt_opsDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "paddle/infrt/dialect/infrt/infrt_opsTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "paddle/infrt/dialect/infrt/infrt_opsAttributes.h.inc"

#define GET_OP_CLASSES
#include "paddle/infrt/dialect/infrt/infrt_ops.h.inc"
