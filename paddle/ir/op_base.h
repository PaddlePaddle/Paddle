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

#include "paddle/ir/operation.h"

namespace ir {
class OpBase {
 public:
  Operation *operation() { return operation_; }

  explicit operator bool() { return operation() != nullptr; }

  operator Operation *() const { return operation_; }

  Operation *operator->() const { return operation_; }

 protected:
  explicit OpBase(Operation *operation) : operation_(operation) {}

 private:
  Operation *operation_;
};

///
/// \brief OpTrait
///
template <class ConcreteTrait>
class OpTraitBase : public OpBase {
 public:
  OpTraitBase::OpTraitBase(Operation *op) : OpBase(op) {}

  static TypeId GetTraitId() { return TypeId::get<ConcreteTrait>(); }
};

class ReadOnlyTrait : public OpTraitBase<ReadOnlyTrait> {
 public:
  explicit ReadOnlyTrait(Operation *op) : OpTraitBase<ReadOnlyTrait>(op) {}
};

///
/// \brief OpInterface
///
template <typename ConcreteInterface>
class OpInterfaceBase : public OpBase {
 public:
  using Concept = typename ConcreteInterface::Concept;

  OpInterfaceBase(Operation *op, Concept *impl) : OpBase(op), impl_(impl) {}

  Concept *impl() { return impl_; }

  static TypeId GetInterfaceId() { return TypeId::get<ConcreteInterface>(); }

 protected:
  Concept *impl_;
};

class InferShapeInterface : public OpInterfaceBase<InferShapeInterface> {
 public:
  struct Concept {
    explicit Concept(void (*infer_shape)(Operation *))
        : infer_shape_(infer_shape) {}
    void (*infer_shape_)(Operation *);
  };

  template <class ConcreteOp>
  struct Model : public Concept {
    static void InferShape(Operation *op) {
      ConcreteOp concret_op = ir::dyn_cast<ConcreteOp>(op);
      if (concret_op == nullptr) throw("concret_op is nullptr");
      concret_op.InferShape();
    }

    Model() : Concept(InferShape) {
      if (sizeof(Model) != sizeof(Concept)) {
        throw("sizeof(Model) != sizeof(Concept)")
      }
    }
  };

  InferShapeInterface(Operation *op, Concept *impl)
      : OpInterfaceBase(op, impl) {}

  void InferShape() { impl_->infer_shape_(operation()); }
};

}  // namespace ir
