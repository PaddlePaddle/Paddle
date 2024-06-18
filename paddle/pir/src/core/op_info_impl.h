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

#include <algorithm>
#include <initializer_list>
#include <set>
#include <string>
#include <utility>

#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/op_base.h"
#include "paddle/pir/include/core/type.h"

namespace pir {
class Dialect;
class InterfaceValue;

///
/// \brief OpInfoImpl class.
///
class OpInfoImpl {
 public:
  ///
  /// \brief Construct and Deconstruct OpInfoImpl. The memory layout of
  /// OpInfoImpl is: std::pair<TypeId, void *>... | TypeId... | OpInfoImpl
  ///
  static OpInfo Create(Dialect *dialect,
                       TypeId op_id,
                       const char *op_name,
                       std::set<InterfaceValue> &&interface_set,
                       const std::vector<TypeId> &trait_set,
                       size_t attributes_num,
                       const char *attributes_name[],
                       VerifyPtr verify_sig,
                       VerifyPtr verify_region);
  static void Destroy(OpInfo info);

  TypeId id() const { return op_id_; }

  Dialect *dialect() const { return dialect_; }

  VerifyPtr VerifySig() const { return verify_sig_; }

  VerifyPtr VerifyRegion() const { return verify_region_; }

  IrContext *ir_context() const;

  /// \brief Search methods for Trait or Interface.
  bool HasTrait(TypeId trait_id) const;

  bool HasInterface(TypeId interface_id) const;

  void AttachInterface(InterfaceValue &&interface_value);

  void *GetInterfaceImpl(TypeId interface_id) const;

  const char *name() const { return op_name_; }

  uint32_t AttributeNum() const { return num_attributes_; }

  const char *GetAttributeByIndex(size_t idx) const {
    return idx < num_attributes_ ? p_attributes_[idx] : nullptr;
  }

  std::vector<std::string> GetAttributesName() const;

 private:
  OpInfoImpl(std::set<InterfaceValue> &&interface_set,
             pir::Dialect *dialect,
             TypeId op_id,
             const char *op_name,
             uint32_t num_traits,
             uint32_t num_attributes,
             const char **p_attributes,
             VerifyPtr verify_sig,
             VerifyPtr verify_region);
  ~OpInfoImpl() = default;
  void Destroy();

  std::set<InterfaceValue> interface_set_;

  /// The dialect of this Op belong to.
  Dialect *dialect_;

  /// The TypeId of this Op.
  TypeId op_id_;

  /// The name of this Op.
  const char *op_name_;

  /// Trait will be recorded by TypeId.
  uint32_t num_traits_ = 0;

  /// The number of attributes for this Op.
  uint32_t num_attributes_ = 0;

  /// Attributes array address.
  const char **p_attributes_{nullptr};

  VerifyPtr verify_sig_{nullptr};

  VerifyPtr verify_region_{nullptr};
};

}  // namespace pir
