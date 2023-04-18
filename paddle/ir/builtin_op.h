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

#include "paddle/ir/op_base.h"

namespace ir {

#define GET_BUILT_IN_OP_LIST ir::FakeConvOp

class ReadOnlyTrait : public OpTraitBase<ReadOnlyTrait> {
 public:
  explicit ReadOnlyTrait(Operation *op) : OpTraitBase<ReadOnlyTrait>(op) {}
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
      ConcreteOp concret_op = ConcreteOp(op);
      if (concret_op == nullptr) throw("concret_op is nullptr");
      concret_op.InferShape();
    }

    Model() : Concept(InferShape) {
      if (sizeof(Model) != sizeof(Concept)) {
        throw("sizeof(Model) != sizeof(Concept)");
      }
    }
  };

  InferShapeInterface(Operation *op, Concept *impl)
      : OpInterfaceBase<InferShapeInterface>(op), impl_(impl) {}

  void InferShape() { impl_->infer_shape_(operation()); }

 private:
  Concept *impl_;
};

class FakeConvOp : public Op<FakeConvOp, ReadOnlyTrait, InferShapeInterface> {
 public:
  using Op::Op;

  ///
  /// \brief Op name.
  ///
  static const char *name() { return "fake_conv"; }

  ///
  /// \brief Op attributes name.
  ///
  static const char *attributes_name_[];

  ///
  /// \brief This Op definition template parameter contains InterfaceInterface,
  /// therefore it must define the InferShape function.
  ///
  static void InferShape() {
    std::cout << "this is a fake conv op interface" << std::endl;
  }
};

const char *FakeConvOp::attributes_name_[] = {"strides",
                                              "paddings",
                                              "padding_algorithm",
                                              "dilations",
                                              "groups",
                                              "data_format"};

}  // namespace ir
