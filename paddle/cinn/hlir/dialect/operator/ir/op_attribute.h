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

#pragma once
#include "paddle/cinn/hlir/dialect/operator/ir/attribute_storage.h"
#include "paddle/cinn/operator_fusion/fusion_tracker/tracker.h"
#include "paddle/pir/include/core/attribute_base.h"

namespace cinn {
namespace dialect {
class FusionTrackerPtrAttribute : public pir::Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(FusionTrackerPtrAttribute,
                                    FusionTrackerPtrAttributeStorage);

  bool operator<(const FusionTrackerPtrAttribute& right) const {
    return storage() < right.storage();
  }

  static std::string name() { return "fusion_tracker"; }

  const cinn::fusion::FusionTrackerPtr& data() const;
};

class GroupInfoAttribute : public pir::Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(GroupInfoAttribute,
                                    GroupInfoAttributeStorage);

  bool operator<(const GroupInfoAttribute& right) const {
    return storage() < right.storage();
  }

  static std::string name() { return "a_group_info"; }

  const GroupInfo& data() const;
};

class CINNKernelInfoAttribute : public pir::Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(CINNKernelInfoAttribute,
                                    CINNKernelInfoAttributeStorage);

  bool operator<(const CINNKernelInfoAttribute& right) const {
    return storage() < right.storage();
  }

  static std::string name() { return "a_cinn_kernel_info"; }

  const cinn::hlir::framework::pir::CINNKernelInfo& data() const;
};

}  // namespace dialect
}  // namespace cinn

IR_DECLARE_EXPLICIT_TYPE_ID(cinn::dialect::GroupInfoAttribute)
IR_DECLARE_EXPLICIT_TYPE_ID(cinn::dialect::CINNKernelInfoAttribute)
IR_DECLARE_EXPLICIT_TYPE_ID(cinn::dialect::FusionTrackerPtrAttribute)
