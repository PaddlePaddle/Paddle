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

#include "paddle/ir/builtin_attribute.h"
#include "paddle/ir/op_info.h"
#include "paddle/ir/type.h"
#include "paddle/ir/value_impl.h"

namespace ir {
template <class ConcreteTrait>
class OpTraitBase;
template <typename ConcreteInterface>
class OpInterfaceBase;

class alignas(8) Operation final {
 public:
  ///
  /// \brief Malloc memory and construct objects in the following order:
  /// OpResultImpls|Operation|OpOperandImpls.
  ///
  static Operation *create(const std::vector<ir::OpResult> &inputs,
                           const std::vector<ir::Type> &output_types,
                           ir::DictionaryAttribute attribute,
                           ir::OpInfo op_info);

  void destroy();

  ir::OpResult GetResultByIndex(uint32_t index);

  std::string print();

  ir::DictionaryAttribute attribute() const { return attribute_; }

  ir::OpInfo op_info() const { return op_info_; }

  uint32_t num_results() const { return num_results_; }

  uint32_t num_operands() const { return num_operands_; }

  template <typename T>
  T dyn_cast() const {
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

 private:
  Operation(uint32_t num_results,
            uint32_t num_operands,
            ir::DictionaryAttribute attribute,
            ir::OpInfo op_info);

  template <typename T, typename Enabler = void>
  struct CastUtil {
    static T call(const Operation *op) {
      throw("Can't dyn_cast to T, T should be a Trait or Interface");
    }
  };
  template <typename T>
  struct CastUtil<T,
                  typename std::enable_if<
                      std::is_base_of<OpTraitBase<T>, T>::value>::type> {
    static T call(const Operation *op) { return T(op); }
  };
  template <typename T>
  struct CastUtil<T,
                  typename std::enable_if<
                      std::is_base_of<OpInterfaceBase<T>, T>::value>::type> {
    static T call(const Operation *op) {
      return T(op, op->op_info_.impl()->GetInterfaceImpl<T>());
    }
  };

  ir::DictionaryAttribute attribute_;

  ir::OpInfo op_info_;

  uint32_t num_results_ = 0;

  uint32_t num_operands_ = 0;
};

}  // namespace ir
