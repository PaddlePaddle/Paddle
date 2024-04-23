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

#include "paddle/cinn/frontend/paddle/pb/block_desc.h"
#include "paddle/common/enforce.h"

namespace cinn::frontend::paddle::pb {

template <>
framework_proto::VarDesc* BlockDesc::GetVar<framework_proto::VarDesc>(
    int32_t idx) {
  PADDLE_ENFORCE_LT(
      idx,
      VarsSize(),
      phi::errors::InvalidArgument(
          "The value of idx and vars.size() is incorrect."
          "Expected idx < vars.size(), but receive idx >= vars.size()."));
  return desc_->mutable_vars(idx);
}

template <>
framework_proto::VarDesc* BlockDesc::AddVar<framework_proto::VarDesc>() {
  return desc_->add_vars();
}

template <>
framework_proto::OpDesc* BlockDesc::GetOp<framework_proto::OpDesc>(
    int32_t idx) {
  PADDLE_ENFORCE_LT(
      idx,
      OpsSize(),
      phi::errors::InvalidArgument(
          "The value of idx and ops.size() is incorrect."
          "Expected idx < ops.size(), but receive idx >= ops.size()."));
  return desc_->mutable_ops(idx);
}

template <>
framework_proto::OpDesc* BlockDesc::AddOp<framework_proto::OpDesc>() {
  return desc_->add_ops();
}

}  // namespace cinn::frontend::paddle::pb
