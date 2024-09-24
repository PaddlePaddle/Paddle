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

#include <glog/logging.h>

#include "paddle/pir/include/core/builder.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/op_base.h"
#include "paddle/pir/include/core/op_trait.h"
#include "paddle/pir/include/core/operation_utils.h"
#include "test/cpp/pir/tools/macros_utils.h"
#include "test/cpp/pir/tools/test_interface.h"
#include "test/cpp/pir/tools/test_trait.h"

namespace test {
///
/// \brief TestRegionOp
///
class RegionOp : public pir::Op<RegionOp, OneRegionTrait> {
 public:
  using Op::Op;
  static const char *name() { return "test.region"; }
  static constexpr uint32_t attributes_num = 0;
  static constexpr const char **attributes_name = nullptr;
  static void Build(pir::Builder &builder,              // NOLINT
                    pir::OperationArgument &argument);  // NOLINT
  void VerifySig() const {}
};

///
/// \brief TestBranchOp
///
class BranchOp : public pir::Op<BranchOp> {
 public:
  using Op::Op;
  static const char *name() { return "test.branch"; }
  static constexpr uint32_t attributes_num = 0;
  static constexpr const char **attributes_name = nullptr;
  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    const std::vector<pir::Value> &target_operands,
                    pir::Block *target);
  void VerifySig() const;
};

// Define case op1.
class Operation1 : public pir::Op<Operation1> {
 public:
  using Op::Op;
  static const char *name() { return "test.operation1"; }
  static constexpr uint32_t attributes_num = 2;
  static const char *attributes_name[attributes_num];
  static void Build(pir::Builder &builder,              // NOLINT
                    pir::OperationArgument &argument);  // NOLINT
  void VerifySig() const;
};

// Define op2.
class Operation2
    : public pir::Op<Operation2, ReadOnlyTrait, InferShapeInterface> {
 public:
  using Op::Op;
  static const char *name() { return "test.operation2"; }
  static constexpr uint32_t attributes_num = 0;
  static constexpr const char **attributes_name = nullptr;
  static void Build(pir::Builder &builder,                // NOLINT
                    pir::OperationArgument &argument) {}  // NOLINT
  void VerifySig() const {}
  static void InferShape() { VLOG(2) << "This is op2's InferShape interface."; }
};

// Define TraitExampleOp.
class TraitExampleOp
    : public pir::Op<TraitExampleOp,
                     pir::SameOperandsShapeTrait,
                     pir::SameOperandsAndResultShapeTrait,
                     pir::SameOperandsElementTypeTrait,
                     pir::SameOperandsAndResultElementTypeTrait,
                     pir::SameOperandsAndResultTypeTrait,
                     pir::SameTypeOperandsTrait> {
 public:
  using Op::Op;
  static const char *name() { return "test.trait_example_op"; }
  static constexpr uint32_t attributes_num = 0;
  static constexpr const char **attributes_name = nullptr;
  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::Value l_operand,
                    pir::Value r_operand,
                    pir::Type out_type);
  void VerifySig() const {}
};

// Define SameOperandsShapeTraitOp1.
class SameOperandsShapeTraitOp1
    : public pir::Op<SameOperandsShapeTraitOp1, pir::SameOperandsShapeTrait> {
 public:
  using Op::Op;
  static const char *name() { return "test.same_operands_shape_op1"; }
  static constexpr uint32_t attributes_num = 0;
  static constexpr const char **attributes_name = nullptr;
  static void Build(pir::Builder &builder,                // NOLINT
                    pir::OperationArgument &argument) {}  // NOLINT
  void VerifySig() const {}
};

// Define SameOperandsShapeTraitOp2.
class SameOperandsShapeTraitOp2
    : public pir::Op<SameOperandsShapeTraitOp2, pir::SameOperandsShapeTrait> {
 public:
  using Op::Op;
  static const char *name() { return "test.same_operands_shape_op2"; }
  static constexpr uint32_t attributes_num = 0;
  static constexpr const char **attributes_name = nullptr;
  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::Value l_operand,
                    pir::Value r_operand,
                    pir::Type out_type);
  void VerifySig() const {}
};

// Define SameOperandsAndResultShapeTraitOp1.
class SameOperandsAndResultShapeTraitOp1
    : public pir::Op<SameOperandsAndResultShapeTraitOp1,
                     pir::SameOperandsAndResultShapeTrait> {
 public:
  using Op::Op;
  static const char *name() {
    return "test.same_operands_and_result_shape_op1";
  }
  static constexpr uint32_t attributes_num = 0;
  static constexpr const char **attributes_name = nullptr;
  static void Build(pir::Builder &builder,                // NOLINT
                    pir::OperationArgument &argument) {}  // NOLINT
  void VerifySig() const {}
};

// Define SameOperandsAndResultShapeTraitOp2.
class SameOperandsAndResultShapeTraitOp2
    : public pir::Op<SameOperandsAndResultShapeTraitOp2,
                     pir::SameOperandsAndResultShapeTrait> {
 public:
  using Op::Op;
  static const char *name() {
    return "test.same_operands_and_result_shape_op2";
  }
  static constexpr uint32_t attributes_num = 0;
  static constexpr const char **attributes_name = nullptr;
  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::Value l_operand,
                    pir::Value r_operand);
  void VerifySig() const {}
};

// Define SameOperandsAndResultShapeTraitOp3.
class SameOperandsAndResultShapeTraitOp3
    : public pir::Op<SameOperandsAndResultShapeTraitOp3,
                     pir::SameOperandsAndResultShapeTrait> {
 public:
  using Op::Op;
  static const char *name() {
    return "test.same_operands_and_result_shape_op3";
  }
  static constexpr uint32_t attributes_num = 0;
  static constexpr const char **attributes_name = nullptr;
  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::Value l_operand,
                    pir::Value r_operand,
                    pir::Type out_type);
  void VerifySig() const {}
};

// Define SameOperandsElementTypeTraitOp1.
class SameOperandsElementTypeTraitOp1
    : public pir::Op<SameOperandsElementTypeTraitOp1,
                     pir::SameOperandsElementTypeTrait> {
 public:
  using Op::Op;
  static const char *name() { return "test.same_operands_element_type_op1"; }
  static constexpr uint32_t attributes_num = 0;
  static constexpr const char **attributes_name = nullptr;
  static void Build(pir::Builder &builder,                // NOLINT
                    pir::OperationArgument &argument) {}  // NOLINT
  void VerifySig() const {}
};

// Define SameOperandsElementTypeTraitOp2.
class SameOperandsElementTypeTraitOp2
    : public pir::Op<SameOperandsElementTypeTraitOp2,
                     pir::SameOperandsElementTypeTrait> {
 public:
  using Op::Op;
  static const char *name() { return "test.same_operands_element_type_op1"; }
  static constexpr uint32_t attributes_num = 0;
  static constexpr const char **attributes_name = nullptr;
  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::Value l_operand,
                    pir::Value r_operand,
                    pir::Type out_type);
  void VerifySig() const {}
};

// Define SameOperandsAndResultElementTypeTraitOp1.
class SameOperandsAndResultElementTypeTraitOp1
    : public pir::Op<SameOperandsAndResultElementTypeTraitOp1,
                     pir::SameOperandsAndResultElementTypeTrait> {
 public:
  using Op::Op;
  static const char *name() {
    return "test.same_operands_and_result_element_type_op1";
  }
  static constexpr uint32_t attributes_num = 0;
  static constexpr const char **attributes_name = nullptr;
  static void Build(pir::Builder &builder,                // NOLINT
                    pir::OperationArgument &argument) {}  // NOLINT
  void VerifySig() const {}
};

// Define SameOperandsAndResultElementTypeTraitOp2.
class SameOperandsAndResultElementTypeTraitOp2
    : public pir::Op<SameOperandsAndResultElementTypeTraitOp2,
                     pir::SameOperandsAndResultElementTypeTrait> {
 public:
  using Op::Op;
  static const char *name() {
    return "test.same_operands_and_result_element_type_op2";
  }
  static constexpr uint32_t attributes_num = 0;
  static constexpr const char **attributes_name = nullptr;
  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::Value l_operand,
                    pir::Value r_operand);
  void VerifySig() const {}
};

// Define SameOperandsAndResultElementTypeTraitOp3.
class SameOperandsAndResultElementTypeTraitOp3
    : public pir::Op<SameOperandsAndResultElementTypeTraitOp3,
                     pir::SameOperandsAndResultElementTypeTrait> {
 public:
  using Op::Op;
  static const char *name() {
    return "test.same_operands_and_result_element_type_op3";
  }
  static constexpr uint32_t attributes_num = 0;
  static constexpr const char **attributes_name = nullptr;
  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::Value l_operand,
                    pir::Value r_operand,
                    pir::Type out_type1,
                    pir::Type out_type2);
  void VerifySig() const {}
};

// Define SameOperandsAndResultTypeTraitOp1.
class SameOperandsAndResultTypeTraitOp1
    : public pir::Op<SameOperandsAndResultTypeTraitOp1,
                     pir::SameOperandsAndResultTypeTrait> {
 public:
  using Op::Op;
  static const char *name() { return "test.same_operands_and_result_type_op1"; }
  static constexpr uint32_t attributes_num = 0;
  static constexpr const char **attributes_name = nullptr;
  static void Build(pir::Builder &builder,                // NOLINT
                    pir::OperationArgument &argument) {}  // NOLINT
  void VerifySig() const {}
};

// Define SameOperandsAndResultTypeTraitOp2.
class SameOperandsAndResultTypeTraitOp2
    : public pir::Op<SameOperandsAndResultTypeTraitOp2,
                     pir::SameOperandsAndResultTypeTrait> {
 public:
  using Op::Op;
  static const char *name() { return "test.same_operands_and_result_type_op2"; }
  static constexpr uint32_t attributes_num = 0;
  static constexpr const char **attributes_name = nullptr;
  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::Value l_operand,
                    pir::Value r_operand);
  void VerifySig() const {}
};

// Define SameOperandsAndResultTypeTraitOp3.
class SameOperandsAndResultTypeTraitOp3
    : public pir::Op<SameOperandsAndResultTypeTraitOp3,
                     pir::SameOperandsAndResultTypeTrait> {
 public:
  using Op::Op;
  static const char *name() { return "test.same_operands_and_result_type_op3"; }
  static constexpr uint32_t attributes_num = 0;
  static constexpr const char **attributes_name = nullptr;

  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::Value l_operand,
                    pir::Value r_operand,
                    pir::Type out_type1,
                    pir::Type out_type2);

  void VerifySig() const {}
};

}  // namespace test

IR_DECLARE_EXPLICIT_TEST_TYPE_ID(test::RegionOp)
IR_DECLARE_EXPLICIT_TEST_TYPE_ID(test::BranchOp)
IR_DECLARE_EXPLICIT_TEST_TYPE_ID(test::Operation1)
IR_DECLARE_EXPLICIT_TEST_TYPE_ID(test::Operation2)
IR_DECLARE_EXPLICIT_TEST_TYPE_ID(test::TraitExampleOp)
IR_DECLARE_EXPLICIT_TEST_TYPE_ID(test::SameOperandsShapeTraitOp1)
IR_DECLARE_EXPLICIT_TEST_TYPE_ID(test::SameOperandsShapeTraitOp2)
IR_DECLARE_EXPLICIT_TEST_TYPE_ID(test::SameOperandsAndResultShapeTraitOp1)
IR_DECLARE_EXPLICIT_TEST_TYPE_ID(test::SameOperandsAndResultShapeTraitOp2)
IR_DECLARE_EXPLICIT_TEST_TYPE_ID(test::SameOperandsAndResultShapeTraitOp3)
IR_DECLARE_EXPLICIT_TEST_TYPE_ID(test::SameOperandsElementTypeTraitOp1)
IR_DECLARE_EXPLICIT_TEST_TYPE_ID(test::SameOperandsElementTypeTraitOp2)
IR_DECLARE_EXPLICIT_TEST_TYPE_ID(test::SameOperandsAndResultElementTypeTraitOp1)
IR_DECLARE_EXPLICIT_TEST_TYPE_ID(test::SameOperandsAndResultElementTypeTraitOp2)
IR_DECLARE_EXPLICIT_TEST_TYPE_ID(test::SameOperandsAndResultElementTypeTraitOp3)
IR_DECLARE_EXPLICIT_TEST_TYPE_ID(test::SameOperandsAndResultTypeTraitOp1)
IR_DECLARE_EXPLICIT_TEST_TYPE_ID(test::SameOperandsAndResultTypeTraitOp2)
IR_DECLARE_EXPLICIT_TEST_TYPE_ID(test::SameOperandsAndResultTypeTraitOp3)

namespace test1 {
// Define case op1.
class Operation1 : public pir::Op<Operation1> {
 public:
  using Op::Op;
  static const char *name() { return "test1.operation1"; }
  static constexpr uint32_t attributes_num = 2;
  static const char *attributes_name[attributes_num];
  static void Build(pir::Builder &builder,              // NOLINT
                    pir::OperationArgument &argument);  // NOLINT
  void VerifySig() const;
};

// Define op2.
class Operation2 : public pir::Op<Operation2> {
 public:
  using Op::Op;
  static const char *name() { return "test1.operation2"; }
  static constexpr uint32_t attributes_num = 0;
  static constexpr const char **attributes_name = nullptr;
  static void Build(pir::Builder &builder,              // NOLINT
                    pir::OperationArgument &argument);  // NOLINT
  void VerifySig() const {}
  static void InferShape() { VLOG(2) << "This is op2's InferShape interface."; }
};

// Define op3.
class Operation3 : public pir::Op<Operation3> {
 public:
  using Op::Op;
  static const char *name() { return "test1.operation3"; }
  static constexpr uint32_t attributes_num = 0;
  static constexpr const char **attributes_name = nullptr;
  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::Value l_operand,
                    pir::Value r_operand);
  void VerifySig() const {}
  static void InferShape() { VLOG(2) << "This is op3's InferShape interface."; }
};

// Define op4.
class Operation4 : public pir::Op<Operation4> {
 public:
  using Op::Op;
  static const char *name() { return "test1.operation4"; }
  static constexpr uint32_t attributes_num = 0;
  static constexpr const char **attributes_name = nullptr;
  static void Build(pir::Builder &builder,              // NOLINT
                    pir::OperationArgument &argument);  // NOLINT
  void VerifySig() const {}
};
}  // namespace test1
IR_DECLARE_EXPLICIT_TEST_TYPE_ID(test1::Operation1)
IR_DECLARE_EXPLICIT_TEST_TYPE_ID(test1::Operation2)
IR_DECLARE_EXPLICIT_TEST_TYPE_ID(test1::Operation3)
IR_DECLARE_EXPLICIT_TEST_TYPE_ID(test1::Operation4)
