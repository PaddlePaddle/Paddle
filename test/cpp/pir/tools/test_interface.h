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

#include "paddle/common/enforce.h"
#include "paddle/pir/include/core/block.h"
#include "paddle/pir/include/core/builder.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/dialect.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/ir_printer.h"
#include "paddle/pir/include/core/op_base.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/core/region.h"
#include "test/cpp/pir/tools/macros_utils.h"

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
      ConcreteOp concrete_op = ConcreteOp(op);
      if (concrete_op == nullptr) throw("concrete_op is nullptr");
      concrete_op.InferShape();
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
IR_DECLARE_EXPLICIT_TEST_TYPE_ID(test::InferShapeInterface)
