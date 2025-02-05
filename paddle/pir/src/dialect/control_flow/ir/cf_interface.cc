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

#include "paddle/pir/include/dialect/control_flow/ir/cf_interface.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"

namespace pir {
TuplePushOp ContainerOpInterface::tuple_push_op() {
  auto value = inlet();
  PADDLE_ENFORCE_EQ(
      value.HasOneUse(),
      true,
      common::errors::InvalidArgument(
          "The inlet value of container op can only be used once."));
  return value.first_use().owner()->dyn_cast<TuplePushOp>();
}
TuplePopOp ContainerOpInterface::tuple_pop_op() {
  auto value = outlet();
  PADDLE_ENFORCE_EQ(
      value.HasOneUse(),
      true,
      common::errors::InvalidArgument(
          "The outlet value of container op can only be used once."));
  return value.first_use().owner()->dyn_cast<TuplePopOp>();
}

}  // namespace pir

IR_DEFINE_EXPLICIT_TYPE_ID(pir::ContainerOpInterface)
