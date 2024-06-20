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

#include "paddle/cinn/frontend/paddle/cpp/block_desc.h"
#include "paddle/common/enforce.h"

namespace cinn::frontend::paddle::cpp {

template <>
VarDesc* BlockDesc::GetVar<VarDesc>(int32_t idx) {
  PADDLE_ENFORCE_LT(
      idx,
      VarsSize(),
      phi::errors::InvalidArgument(
          "The value of idx and vars.size() is incorrect."
          "Expected idx < vars.size(), but receive idx >= vars.size()."));
  return &vars_[idx];
}

template <>
const VarDesc& BlockDesc::GetConstVar<VarDesc>(int32_t idx) const {
  PADDLE_ENFORCE_LT(
      idx,
      static_cast<int32_t>(VarsSize()),
      phi::errors::InvalidArgument(
          "The value of idx and vars.size() is incorrect."
          "Expected idx < vars.size(), but receive idx >= vars.size()."));
  return vars_[idx];
}

template <>
VarDesc* BlockDesc::AddVar<VarDesc>() {
  vars_.emplace_back();
  return &vars_.back();
}

template <>
OpDesc* BlockDesc::GetOp<OpDesc>(int32_t idx) {
  PADDLE_ENFORCE_LT(
      idx,
      OpsSize(),
      phi::errors::InvalidArgument(
          "The value of idx and ops.size() is incorrect."
          "Expected idx < ops.size(), but receive idx >= ops.size()."));
  return &ops_[idx];
}

template <>
const OpDesc& BlockDesc::GetConstOp<OpDesc>(int32_t idx) const {
  PADDLE_ENFORCE_LT(
      idx,
      static_cast<int32_t>(OpsSize()),
      phi::errors::InvalidArgument(
          "The value of idx and ops.size() is incorrect."
          "Expected idx < ops.size(), but receive idx >= ops.size()."));
  return ops_[idx];
}

template <>
OpDesc* BlockDesc::AddOp<OpDesc>() {
  ops_.emplace_back();
  return &ops_.back();
}

}  // namespace cinn::frontend::paddle::cpp
