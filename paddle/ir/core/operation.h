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
#include <vector>
#include "paddle/ir/core/block.h"
#include "paddle/ir/core/enforce.h"
#include "paddle/ir/core/macros.h"
#include "paddle/ir/core/op_info.h"
#include "paddle/ir/core/operation_utils.h"
#include "paddle/ir/core/type.h"

namespace ir {
class OpBase;
class Program;
class OpOperand;
class OpResult;

namespace detial {
class BlockOperandImpl;
}  // namespace detial

class IR_API alignas(8) Operation final {
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
                           size_t num_regions = 0,
                           const std::vector<Block *> &successors = {});
  static Operation *Create(OperationArgument &&op_argument);

  ///
  /// \brief Destroy the operation objects and free memory by create().
  ///
  void Destroy();

  IrContext *ir_context() const;

  Dialect *dialect() const;

  OpResult result(uint32_t index) const;

  OpOperand operand(uint32_t index) const;

  Value operand_source(uint32_t index) const;

  uint32_t num_successors() const { return num_successors_; }
  BlockOperand block_operand(uint32_t index) const;
  Block *successor(uint32_t index) const;
  void set_successor(Block *block, unsigned index);
  bool HasSuccessors() { return num_successors_ != 0; }

  /// Returns the region held by this operation at position 'index'.
  Region &region(unsigned index);
  const Region &region(unsigned index) const;
  uint32_t num_regions() const { return num_regions_; }

  void Print(std::ostream &os);

  const AttributeMap &attributes() const { return attributes_; }

  template <typename T>
  T attribute(const std::string &name) {
    Attribute attr = attribute(name);
    IR_ENFORCE(attr.isa<T>(), "Attribute (%s) type is not right.", name);
    return attr.dyn_cast<T>();
  }

  void set_attribute(const std::string &key, Attribute value) {
    attributes_[key] = value;
  }

  Attribute attribute(const std::string &key) const;

  bool HasAttribute(const std::string &key) const {
    return attributes_.find(key) != attributes_.end();
  }

  ir::OpInfo info() const { return info_; }

  uint32_t num_results() const { return num_results_; }

  uint32_t num_operands() const { return num_operands_; }

  std::string name() const;

  template <typename T>
  T dyn_cast() {
    return CastUtil<T>::call(this);
  }

  template <typename T>
  bool isa() const {
    return T::classof(this);
  }

  template <typename Trait>
  bool HasTrait() const {
    return info_.HasTrait<Trait>();
  }

  template <typename Interface>
  bool HasInterface() const {
    return info_.HasInterface<Interface>();
  }

  const Block *GetParent() const { return parent_; }

  Block *GetParent() {
    return const_cast<Block *>(
        const_cast<const Operation *>(this)->GetParent());
  }

  Region *GetParentRegion();

  Operation *GetParentOp() const;

  const Program *GetParentProgram() const;

  Program *GetParentProgram() {
    return const_cast<Program *>(
        const_cast<const Operation *>(this)->GetParentProgram());
  }

  operator Block::iterator() { return position_; }

  operator Block::const_iterator() const { return position_; }

  /// Replace all uses of results of this operation with the provided 'values'.
  void ReplaceAllUsesWith(const std::vector<Value> &values);

  void ReplaceAllUsesWith(const std::vector<OpResult> &op_results);

  inline void ReplaceAllUsesWith(Value value) {
    ReplaceAllUsesWith(std::vector<Value>{value});
  }

  void Verify();

  std::vector<OpOperand> operands() const;

  std::vector<OpResult> results() const;

 private:
  DISABLE_COPY_AND_ASSIGN(Operation);
  Operation(const AttributeMap &attribute,
            ir::OpInfo op_info,
            uint32_t num_results,
            uint32_t num_operands,
            uint32_t num_regions,
            uint32_t num_successors);

  template <typename T, typename Enabler = void>
  struct CastUtil {
    static T call(Operation *op) {
      throw("Can't dyn_cast to T, T should be a Op or Trait or Interface");
    }
  };

  // Allow access to 'SetParent'.
  friend class Block;
  void SetParent(Block *parent, const Block::iterator &position);

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
  const uint32_t num_successors_ = 0;

  detail::BlockOperandImpl *block_operands_{nullptr};
  Region *regions_{nullptr};
  Block *parent_{nullptr};
  Block::iterator position_;
};

}  // namespace ir
