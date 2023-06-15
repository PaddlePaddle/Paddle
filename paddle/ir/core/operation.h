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

#include <iostream>
#include "paddle/ir/core/op_info.h"
#include "paddle/ir/core/operation_utils.h"
#include "paddle/ir/core/type.h"
#include "paddle/ir/core/value_impl.h"

namespace ir {
class OpBase;
class Program;
class Block;

class alignas(8) Operation final {
 public:
  ///
  /// \brief Malloc memory and construct objects in the following order:
  /// OpResultImpls|Operation|OpOperandImpls.
  /// NOTE: Similar to new and delete, the destroy() and the create() need to be
  /// used in conjunction.
  ///
  static Operation *create(const std::vector<ir::OpResult> &inputs,
                           const AttributeMap &attribute,
                           const std::vector<ir::Type> &output_types,
                           ir::OpInfo op_info,
                           size_t num_regions = 0);
  static Operation *create(OperationArgument &&op_argument);

  ///
  /// \brief Destroy the operation objects and free memory by create().
  ///
  void destroy();

  Block *parent() const { return parent_; }

  IrContext *ir_context() const;

  ir::OpResult GetResultByIndex(uint32_t index) const;

  ir::OpOperand GetOperandByIndex(uint32_t index) const;

  std::string print();

  const AttributeMap &attribute() const { return attribute_; }

  ir::OpInfo op_info() const { return op_info_; }

  uint32_t num_results() const { return num_results_; }

  uint32_t num_operands() const { return num_operands_; }

  uint32_t num_regions() const { return num_regions_; }

  std::string op_name() const;

  template <typename T>
  T dyn_cast() {
    return CastUtil<T>::call(this);
  }

  template <typename Trait>
  bool HasTrait() const {
    return op_info_.HasTrait<Trait>();
  }

  template <typename Interface>
  bool HasInterface() const {
    return op_info_.HasInterface<Interface>();
  }

  Program *parent_program() const { return parent_program_; }

  void set_parent_program(Program *parent_program) {
    parent_program_ = parent_program;
  }

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

  AttributeMap attribute_;

  OpInfo op_info_;

  const uint32_t num_results_ = 0;
  const uint32_t num_operands_ = 0;
  const uint32_t num_regions_ = 0;

  Region *regions_{nullptr};
  Program *parent_program_{nullptr};
  Block *parent_{nullptr};
};

}  // namespace ir
