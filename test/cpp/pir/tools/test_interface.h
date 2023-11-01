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
#include <gtest/gtest.h>
#include <sstream>

#include "paddle/pir/core/block.h"
#include "paddle/pir/core/builder.h"
#include "paddle/pir/core/builtin_attribute.h"
#include "paddle/pir/core/builtin_op.h"
#include "paddle/pir/core/builtin_type.h"
#include "paddle/pir/core/dialect.h"
#include "paddle/pir/core/enforce.h"
#include "paddle/pir/core/ir_context.h"
#include "paddle/pir/core/ir_printer.h"
#include "paddle/pir/core/op_base.h"
#include "paddle/pir/core/program.h"
#include "paddle/pir/core/region.h"

namespace test {
/// \brief Define built-in Interface, derived from OpInterfaceBase. Concepts and
/// Models need to be defined within the class. Concept defines abstract
/// interface functions, and Model is a template class that defines the specific
/// implementation of interface functions based on template parameters.
class InferShapeInterface : public pir::OpInterfaceBase<InferShapeInterface> {
 public:
  struct Concept {
    explicit Concept(void (*infer_shape)(pir::Operation *))
        : infer_shape(infer_shape) {}
    void (*infer_shape)(pir::Operation *);
  };

  template <class ConcreteOp>
  struct Model : public Concept {
    static void InferShape(pir::Operation *op) {
      ConcreteOp concret_op = ConcreteOp(op);
      if (concret_op == nullptr) throw("concret_op is nullptr");
      concret_op.InferShape();
    }

    Model() : Concept(InferShape) {}
  };

  InferShapeInterface(pir::Operation *op, Concept *impl)
      : pir::OpInterfaceBase<InferShapeInterface>(op), impl_(impl) {}

  void InferShape() { impl_->infer_shape(operation()); }

 private:
  Concept *impl_;
};

}  // namespace test
IR_DECLARE_EXPLICIT_TYPE_ID(test::InferShapeInterface)
