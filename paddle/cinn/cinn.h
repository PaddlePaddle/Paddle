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

/**
 * This file exposes some internal APIs to global cinn namespace to make usage
 * more friendly.
 */
#pragma once
#include "paddle/cinn/backends/codegen_c.h"
#include "paddle/cinn/backends/codegen_c_x86.h"
#include "paddle/cinn/common/common.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/lang/builtin.h"
#include "paddle/cinn/lang/compute.h"
#include "paddle/cinn/lang/lower.h"
#include "paddle/cinn/lang/placeholder.h"
#include "paddle/cinn/optim/optimize.h"

namespace cinn {

using ast_gen_ius::TensorGroup;
using backends::CodeGenC;
using backends::CodeGenCX86;
using backends::Outputs;
using ir::Module;
using ir::Var;
using lang::Buffer;
using lang::CallExtern;
using lang::CallLowered;
using lang::Compute;
using lang::LowerToAst;
using lang::Placeholder;
using lang::ReduceAll;
using lang::ReduceAny;
using lang::ReduceMax;
using lang::ReduceMin;
using lang::ReduceMul;
using lang::ReduceSum;
using optim::Optimize;

using lang::logic_and;
using lang::logic_or;

using common::Target;

}  // namespace cinn
