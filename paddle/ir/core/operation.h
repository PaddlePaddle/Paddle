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

#include <ostream>
#include "paddle/ir/core/op_info.h"
#include "paddle/ir/core/operation_utils.h"
#include "paddle/ir/core/type.h"

namespace ir {
class OpBase;
class Program;
class Block;
class OpOperand;
class OpResult;

class alignas(8) Operation final {
 public:
  ///
  /// \brief Malloc memory and construct objects in the following order:
  /// OpResultImpls|Operation|OpOperandImpls.
  /// NOTE: Similar to new and delete, the destroy() and the create() need to be
  /// used in conjunction.
  ///
  static Operation *Create(const std::vector<ir::OpResult> &inputs,
                           const AttributeMap &attributes,
                           const std::vector<ir::Type> &output_types,
                           ir::OpInfo op_info,
                           size_t num_regions = 0);
  static Operation *Create(OperationArgument &&op_argument);

  ///
  /// \brief Destroy the operation objects and free memory by create().
  ///
  void Destroy();

  IrContext *ir_context() const;

  OpResult GetResultByIndex(uint32_t index) const;

  OpOperand GetOperandByIndex(uint32_t index) const;

  void Print(std::ostream &os);

  const AttributeMap &attributes() const { return attributes_; }

  void SetAttribute(const std::string &key, Attribute value) {
    attributes_[key] = value;
  }

  ir::OpInfo info() const { return info_; }

  uint32_t num_results() const { return num_results_; }

  uint32_t num_operands() const { return num_operands_; }

  uint32_t num_regions() const { return num_regions_; }

  std::string name() const;

  template <typename T>
  T dyn_cast() {
    return CastUtil<T>::call(this);
  }

  template <typename Trait>
  bool HasTrait() const {
    return info_.HasTrait<Trait>();
  }

  template <typename Interface>
  bool HasInterface() const {
    return info_.HasInterface<Interface>();
  }

  Block *GetParentBlock() const { return parent_; }

  Region *GetParentRegion() const;

  Operation *GetParentOp() const;

  Program *GetParentProgram();

  /// Returns the region held by this operation at position 'index'.
  Region &GetRegion(unsigned index);

 private:
  Operation(const AttributeMap &attribute,
            ir::OpInfo op_info,
            uint32_t num_results,
            uint32_t num_operands,
            uint32_t num_regions);

  template <typename T, typename Enabler = void>
  struct CastUtil {
    static T call(Operation *op) {
      throw("Can't dyn_cast to T, T should be a Op or Trait or Interface");
    }
  };

  friend class Block;
  void set_parent(Block *parent) { parent_ = parent; }

  template <typename T>
  struct CastUtil<
      T,
      typename std::enable_if<std::is_base_of<OpBase, T>::value>::type> {
    static T call(Operation *op) { return T::dyn_cast(op); }
  };

  AttributeMap attributes_;

  OpInfo info_;

  const uint32_t num_results_ = 0;
  const uint32_t num_operands_ = 0;
  const uint32_t num_regions_ = 0;

  Region *regions_{nullptr};
  Block *parent_{nullptr};
};

}  // namespace ir
