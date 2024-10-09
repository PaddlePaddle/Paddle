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

#include "paddle/pir/include/core/op_base.h"

namespace pir {

class TuplePushOp;
class TuplePopOp;
///
/// \brief This interface marks the op can create a container.
///
class ContainerOpInterface : public OpInterfaceBase<ContainerOpInterface> {
 public:
  struct Concept {
    Value (*container_)(Operation* op);
    Value (*inlet_)(Operation* op);
    Value (*outlet_)(Operation* op);
    size_t (*tuple_size_)(Operation* op);
    Value (*inlet_element_)(Operation* op, size_t index);
    Value (*outlet_element_)(Operation* op, size_t index);
  };

  template <class ConcreteOp>
  struct Model : public Concept {
    Model()
        : Concept{container,
                  inlet,
                  outlet,
                  tuple_size,
                  inlet_element,
                  inlet_element} {}
    static Value container(Operation* op) {
      return op->dyn_cast<ConcreteOp>().container();
    }
    static Value inlet(Operation* op) {
      return op->dyn_cast<ConcreteOp>().inlet();
    }
    static Value outlet(Operation* op) {
      return op->dyn_cast<ConcreteOp>().outlet();
    }
    static size_t tuple_size(Operation* op) {
      return op->dyn_cast<ConcreteOp>().tuple_size();
    }
    static Value inlet_element(Operation* op, size_t index) {
      return op->dyn_cast<ConcreteOp>().container();
    }
    static Value outlet_element(Operation* op, size_t index) {
      return op->dyn_cast<ConcreteOp>().container();
    }
  };

  Value container() { return impl_->container_(operation()); }
  Value inlet() { return impl_->inlet_(operation()); }
  Value outlet() { return impl_->outlet_(operation()); }
  size_t tuple_size() { return impl_->tuple_size_(operation()); }
  Value inlet_element(size_t index) {
    return impl_->inlet_element_(operation(), index);
  }
  Value outlet_element(size_t index) {
    return impl_->outlet_element_(operation(), index);
  }

  TuplePushOp tuple_push_op();
  TuplePopOp tuple_pop_op();
  /// Constructor
  ContainerOpInterface(const pir::Operation* op, Concept* impl)
      : OpInterfaceBase<ContainerOpInterface>(op), impl_(impl) {}

 private:
  Concept* impl_;
};
}  // namespace pir

IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::ContainerOpInterface)
