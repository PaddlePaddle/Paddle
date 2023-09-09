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
#include <sstream>

#include "paddle/ir/core/block.h"
#include "paddle/ir/core/builder.h"
#include "paddle/ir/core/builtin_attribute.h"
#include "paddle/ir/core/builtin_op.h"
#include "paddle/ir/core/builtin_type.h"
#include "paddle/ir/core/dialect.h"
#include "paddle/ir/core/enforce.h"
#include "paddle/ir/core/ir_context.h"
#include "paddle/ir/core/ir_printer.h"
#include "paddle/ir/core/op_base.h"
#include "paddle/ir/core/program.h"
#include "paddle/ir/core/region.h"

/// \brief Define built-in Trait, derived from OpTraitBase.
class ReadOnlyTrait : public ir::OpTraitBase<ReadOnlyTrait> {
 public:
  explicit ReadOnlyTrait(ir::Operation *op)
      : ir::OpTraitBase<ReadOnlyTrait>(op) {}
};
IR_DECLARE_EXPLICIT_TYPE_ID(ReadOnlyTrait)
IR_DEFINE_EXPLICIT_TYPE_ID(ReadOnlyTrait)

/// \brief Define built-in Interface, derived from OpInterfaceBase. Concepts and
/// Models need to be defined within the class. Concept defines abstract
/// interface functions, and Model is a template class that defines the specific
/// implementation of interface functions based on template parameters.
class InferShapeInterface : public ir::OpInterfaceBase<InferShapeInterface> {
 public:
  struct Concept {
    explicit Concept(void (*infer_shape)(ir::Operation *))
        : infer_shape_(infer_shape) {}
    void (*infer_shape_)(ir::Operation *);
  };

  template <class ConcreteOp>
  struct Model : public Concept {
    static void InferShape(ir::Operation *op) {
      ConcreteOp concret_op = ConcreteOp(op);
      if (concret_op == nullptr) throw("concret_op is nullptr");
      concret_op.InferShape();
    }

    Model() : Concept(InferShape) {}
  };

  InferShapeInterface(ir::Operation *op, Concept *impl)
      : ir::OpInterfaceBase<InferShapeInterface>(op), impl_(impl) {}

  void InferShape() { impl_->infer_shape_(operation()); }

 private:
  Concept *impl_;
};
IR_DECLARE_EXPLICIT_TYPE_ID(InferShapeInterface)
IR_DEFINE_EXPLICIT_TYPE_ID(InferShapeInterface)

ir::AttributeMap CreateAttributeMap(std::vector<std::string> attribute_names,
                                    std::vector<std::string> attributes) {
  ir::IrContext *ctx = ir::IrContext::Instance();
  ir::AttributeMap attr_map;
  for (size_t i = 0; i < attribute_names.size(); i++) {
    ir::Attribute attr_value = ir::StrAttribute::get(ctx, attributes[i]);
    attr_map.insert(
        std::pair<std::string, ir::Attribute>(attribute_names[i], attr_value));
  }
  return attr_map;
}

// Define op1.
class Operation1 : public ir::Op<Operation1> {
 public:
  using Op::Op;
  static const char *name() { return "test.operation1"; }
  static constexpr uint32_t attributes_num = 2;
  static const char *attributes_name[attributes_num];  // NOLINT
  void Verify() {
    auto &attributes = this->attributes();
    if (attributes.count("op1_attr1") == 0 ||
        !attributes.at("op1_attr1").isa<ir::StrAttribute>()) {
      throw("Type of attribute: parameter_name is not right.");
    }
    if (attributes.count("op1_attr2") == 0 ||
        !attributes.at("op1_attr2").isa<ir::StrAttribute>()) {
      throw("Type of attribute: parameter_name is not right.");
    }
  }
  static void Build(const ir::Builder &builder,
                    ir::OperationArgument &argument) {  // NOLINT
    std::vector<ir::OpResult> inputs = {};
    std::vector<ir::Type> output_types = {
        ir::Float32Type::get(builder.ir_context())};
    std::unordered_map<std::string, ir::Attribute> attributes =
        CreateAttributeMap({"op1_attr1", "op1_attr2"},
                           {"op1_attr1", "op1_attr2"});
    argument.AddOperands(inputs.begin(), inputs.end());
    argument.AddOutputs(output_types.begin(), output_types.end());
    argument.AddAttributes(attributes.begin(), attributes.end());
  }
};
const char *Operation1::attributes_name[attributes_num] = {  // NOLINT
    "op1_attr1",
    "op1_attr2"};

IR_DECLARE_EXPLICIT_TYPE_ID(Operation1)
IR_DEFINE_EXPLICIT_TYPE_ID(Operation1)

// Define op2.
class Operation2
    : public ir::Op<Operation2, ReadOnlyTrait, InferShapeInterface> {
 public:
  using Op::Op;
  static const char *name() { return "test.operation2"; }
  static constexpr uint32_t attributes_num = 2;
  static const char *attributes_name[attributes_num];  // NOLINT
  void Verify() {
    auto &attributes = this->attributes();
    if (attributes.count("op2_attr1") == 0 ||
        (!attributes.at("op2_attr1").isa<ir::StrAttribute>())) {
      throw("Type of attribute: parameter_name is not right.");
    }
    if (attributes.count("op2_attr2") == 0 ||
        (!attributes.at("op2_attr2").isa<ir::StrAttribute>())) {
      throw("Type of attribute: parameter_name is not right.");
    }
  }
  static void InferShape() { VLOG(2) << "This is op2's InferShape interface."; }
};
const char *Operation2::attributes_name[attributes_num] = {  // NOLINT
    "op2_attr1",
    "op2_attr2"};
IR_DECLARE_EXPLICIT_TYPE_ID(Operation2)
IR_DEFINE_EXPLICIT_TYPE_ID(Operation2)

// Define a dialect, op1 and op2 will be registered by this dialect.
class TestDialect : public ir::Dialect {
 public:
  explicit TestDialect(ir::IrContext *context)
      : ir::Dialect(name(), context, ir::TypeId::get<TestDialect>()) {
    initialize();
  }
  static const char *name() { return "test"; }

  void PrintOperation(ir::Operation *op,
                      ir::IrPrinter &printer) const override {
    printer.PrintOpResult(op);
    printer.os << " =";

    printer.os << " \"" << op->name() << "\"";
    printer.PrintOpOperands(op);
  }

 private:
  void initialize() { RegisterOps<Operation1, Operation2>(); }
};
IR_DECLARE_EXPLICIT_TYPE_ID(TestDialect)
IR_DEFINE_EXPLICIT_TYPE_ID(TestDialect)

TEST(op_test, op_test) {
  // (1) Register Dialect, Operation1, Operation2 into IrContext.
  ir::IrContext *ctx = ir::IrContext::Instance();
  ir::Dialect *test_dialect = ctx->GetOrRegisterDialect<TestDialect>();
  EXPECT_EQ(test_dialect != nullptr, true);

  // (2) Get registered operations.
  std::string op1_name = Operation1::name();
  ir::OpInfo op1_info = ctx->GetRegisteredOpInfo(op1_name);
  EXPECT_TRUE(op1_info);
  std::string op2_name = Operation2::name();
  ir::OpInfo op2_info = ctx->GetRegisteredOpInfo(op2_name);
  EXPECT_TRUE(op2_info);
  EXPECT_EQ(op1_info.HasTrait<ReadOnlyTrait>(), false);
  EXPECT_EQ(op1_info.HasInterface<InferShapeInterface>(), false);
  EXPECT_EQ(op2_info.HasTrait<ReadOnlyTrait>(), true);
  EXPECT_EQ(op2_info.HasInterface<InferShapeInterface>(), true);

  // (3) Test uses for op.
  std::vector<ir::OpResult> op_inputs = {};
  std::vector<ir::Type> op_output_types = {ir::Float32Type::get(ctx)};
  ir::Operation *op2 =
      ir::Operation::Create(op_inputs,
                            CreateAttributeMap({"op2_attr1", "op2_attr2"},
                                               {"op2_attr1", "op2_attr2"}),
                            op_output_types,
                            op2_info);

  ReadOnlyTrait trait = op2->dyn_cast<ReadOnlyTrait>();
  EXPECT_EQ(trait.operation(), op2);
  InferShapeInterface interface = op2->dyn_cast<InferShapeInterface>();
  interface.InferShape();
  Operation2 Op2 = op2->dyn_cast<Operation2>();
  EXPECT_EQ(Op2.operation(), op2);
  op2->Destroy();
}

TEST(op_test, region_test) {
  // (1) Register Dialect, Operation1, Operation2 into IrContext.
  ir::IrContext *ctx = ir::IrContext::Instance();
  ir::Dialect *test_dialect = ctx->GetOrRegisterDialect<TestDialect>();
  EXPECT_EQ(test_dialect != nullptr, true);

  // (2) Get registered operations.
  ir::OpInfo op1_info = ctx->GetRegisteredOpInfo(Operation1::name());
  ir::OpInfo op2_info = ctx->GetRegisteredOpInfo(Operation2::name());

  ir::Operation *op1 =
      ir::Operation::Create({},
                            CreateAttributeMap({"op1_attr1", "op1_attr2"},
                                               {"op1_attr1", "op1_attr2"}),
                            {ir::Float32Type::get(ctx)},
                            op1_info);
  ir::Operation *op1_2 =
      ir::Operation::Create({},
                            CreateAttributeMap({"op1_attr1", "op1_attr2"},
                                               {"op1_attr1", "op1_attr2"}),
                            {ir::Float32Type::get(ctx)},
                            op1_info);

  ir::OperationArgument argument(op2_info);
  argument.attributes = CreateAttributeMap({"op2_attr1", "op2_attr2"},
                                           {"op2_attr1", "op2_attr2"});
  argument.output_types = {ir::Float32Type::get(ctx)};
  argument.num_regions = 1;

  ir::Operation *op3 = ir::Operation::Create(std::move(argument));
  // argument.regions.emplace_back(std::make_unique<ir::Region>());

  ir::Region &region = op3->region(0);
  EXPECT_EQ(region.empty(), true);

  // (3) Test custom operation printer
  std::stringstream ss;
  op1->Print(ss);
  EXPECT_EQ(ss.str(), " (%0) = \"test.operation1\" ()");

  region.push_back(new ir::Block());
  region.push_front(new ir::Block());
  region.insert(region.begin(), new ir::Block());
  ir::Block *block = region.front();
  block->push_front(op1);
  block->insert(block->begin(), op1_2);
  op3->Destroy();
}

TEST(op_test, module_op_death) {
  ir::IrContext *ctx = ir::IrContext::Instance();
  ir::OpInfo op_info = ctx->GetRegisteredOpInfo(ir::ModuleOp::name());

  std::vector<ir::OpResult> inputs{ir::OpResult()};
  ir::AttributeMap attrs{{"program", ir::Int32Attribute::get(ctx, 1)}};
  std::vector<ir::Type> output_types = {ir::Float32Type::get(ctx)};

  EXPECT_THROW(ir::Operation::Create(inputs, {}, {}, op_info),
               ir::IrNotMetException);
  EXPECT_THROW(ir::Operation::Create({}, attrs, {}, op_info),
               ir::IrNotMetException);
  EXPECT_THROW(ir::Operation::Create({}, {}, output_types, op_info),
               ir::IrNotMetException);

  ir::Program program(ctx);

  EXPECT_EQ(program.module_op().program(), &program);
  EXPECT_EQ(program.module_op().ir_context(), ctx);

  program.module_op()->set_attribute("program",
                                     ir::PointerAttribute::get(ctx, &program));
}
