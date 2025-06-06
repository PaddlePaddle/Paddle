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

#include "paddle/ap/include/paddle/hlir/op_dialect.h"
#include "paddle/ap/include/paddle/hlir/manual_op.h"

namespace ap {
namespace dialect {

OperatorDialect::OperatorDialect(::pir::IrContext *context)
    : ::pir::Dialect(
          name(), context, ::pir::TypeId::get<ap::dialect::OperatorDialect>()) {
  this->initialize();
}

void OperatorDialect::initialize() {
  RegisterOp<FacadeOp>();
  RegisterOp<AddOp>();
  RegisterOp<UpSpiderOp>();
  RegisterOp<DownSpiderOp>();
  RegisterOp<LoadFromRegisterOp>();
  RegisterOp<StoreToRegisterOp>();
  RegisterOp<LoadFromGlobalOp>();
  RegisterOp<StoreToGlobalOp>();
}

}  // namespace dialect
}  // namespace ap

IR_DEFINE_EXPLICIT_TYPE_ID(ap::dialect::OperatorDialect)
