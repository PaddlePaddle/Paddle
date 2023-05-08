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

#include <gtest/gtest.h>

#include "paddle/ir/builtin_type.h"
#include "paddle/ir/dialect.h"
#include "paddle/ir/ir_context.h"
#include "paddle/ir/op_base.h"

/// \brief Define built-in Trait, derived from OpTraitBase.
class ReadOnlyTrait : public ir::OpTraitBase<ReadOnlyTrait> {
 public:
  explicit ReadOnlyTrait(const ir::Operation *op)
      : ir::OpTraitBase<ReadOnlyTrait>(op) {}
};

/// \brief Define built-in Interface, derived from OpInterfaceBase. Concepts and
/// Models need to be defined within the class. Concept defines abstract
/// interface functions, and Model is a template class that defines the specific
/// implementation of interface functions based on template parameters.
class InferShapeInterface : public ir::OpInterfaceBase<InferShapeInterface> {
 public:
  struct Concept {
    explicit Concept(void (*infer_shape)(const ir::Operation *))
        : infer_shape_(infer_shape) {}
    void (*infer_shape_)(const ir::Operation *);
  };

  template <class ConcreteOp>
  struct Model : public Concept {
    static void InferShape(const ir::Operation *op) {
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

  InferShapeInterface(const ir::Operation *op, Concept *impl)
      : ir::OpInterfaceBase<InferShapeInterface>(op), impl_(impl) {}

  void InferShape() { impl_->infer_shape_(operation()); }

 private:
  Concept *impl_;
};

// Define op1.
class Operation1 : public ir::Op<Operation1> {
 public:
  using Op::Op;
  static const char *name() { return "Operation1"; }
  static const char *attributes_name_[];
  static uint32_t attributes_num() { return 2; }
};
const char *Operation1::attributes_name_[] = {"op1_attr1", "op1_attr2"};

// Define op2.
class Operation2
    : public ir::Op<Operation2, ReadOnlyTrait, InferShapeInterface> {
 public:
  using Op::Op;
  static const char *name() { return "Operation2"; }
  static const char *attributes_name_[];
  static uint32_t attributes_num() { return 2; }
  static void InferShape() {
    std::cout << "This is op2's InferShape interface." << std::endl;
  }
};
const char *Operation2::attributes_name_[] = {"op2_attr1", "op2_attr2"};

// Define a dialect, op1 and op2 will be registered by this dialect.
class TestDialect : public ir::Dialect {
 public:
  explicit TestDialect(ir::IrContext *context)
      : ir::Dialect(name(), context, ir::TypeId::get<TestDialect>()) {
    initialize();
  }
  static const char *name() { return "op_test"; }

 private:
  void initialize() { RegisterOps<Operation1, Operation2>(); }
};

ir::DictionaryAttribute CreateAttribute(std::string attribute_name,
                                        std::string attribute) {
  ir::IrContext *ctx = ir::IrContext::Instance();
  ir::StrAttribute attr_name = ir::StrAttribute::get(ctx, attribute_name);
  ir::Attribute attr_value = ir::StrAttribute::get(ctx, attribute);
  std::map<ir::StrAttribute, ir::Attribute> named_attr;
  named_attr.insert(
      std::pair<ir::StrAttribute, ir::Attribute>(attr_name, attr_value));
  return ir::DictionaryAttribute::get(ctx, named_attr);
}

TEST(op_test, op_test) {
  // (1) Register Dialect, Operation1, Operation2 into IrContext.
  ir::IrContext *ctx = ir::IrContext::Instance();
  ir::Dialect *test_dialect = ctx->GetOrRegisterDialect<TestDialect>();
  std::cout << test_dialect << std::endl;

  // (2) Get registered operations.
  std::string op1_name =
      test_dialect->name() + "." + std::string(Operation1::name());
  ir::OpInfoImpl *op1_info = ctx->GetRegisteredOpInfo(op1_name);
  EXPECT_EQ(op1_info != nullptr, true);
  std::string op2_name =
      test_dialect->name() + "." + std::string(Operation2::name());
  ir::OpInfoImpl *op2_info = ctx->GetRegisteredOpInfo(op2_name);
  EXPECT_EQ(op2_info != nullptr, true);

  EXPECT_EQ(op1_info->HasTrait<ReadOnlyTrait>(), false);
  EXPECT_EQ(op1_info->HasInterface<InferShapeInterface>(), false);
  EXPECT_EQ(op2_info->HasTrait<ReadOnlyTrait>(), true);
  EXPECT_EQ(op2_info->HasInterface<InferShapeInterface>(), true);

  // (3) Test uses for op.
  std::vector<ir::OpResult> op_inputs = {};
  std::vector<ir::Type> op_output_types = {ir::Float32Type::get(ctx)};
  ir::Operation *op =
      ir::Operation::create(op_inputs,
                            op_output_types,
                            CreateAttribute("op1_name", "op1_attr"),
                            op2_info);

  if (op->HasTrait<ReadOnlyTrait>()) {
    ReadOnlyTrait trait = op->dyn_cast<ReadOnlyTrait>();
    EXPECT_EQ(trait.operation(), op);
  }
  if (op->HasInterface<InferShapeInterface>()) {
    InferShapeInterface interface = op->dyn_cast<InferShapeInterface>();
    interface.InferShape();
  }

  op->destroy();
}
