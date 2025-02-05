/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/infershape_utils.h"

#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "paddle/fluid/framework/attribute.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/compat/op_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/core/kernel_registry.h"

namespace paddle {
namespace framework {

void TestInferMeta(bool bool_attr,
                   int int_attr,
                   int64_t int64_attr,
                   float float_attr,
                   const std::string& str_attr,
                   const std::vector<bool>& vec_bool_attr,
                   const std::vector<int>& vec_int_attr,
                   const std::vector<int64_t>& vec_int64_attr,
                   const std::vector<float>& vec_float_attr,
                   const std::vector<double>& vec_double_attr,
                   const std::vector<std::string>& vec_str_attr) {
  ASSERT_EQ(bool_attr, true);
  ASSERT_EQ(int_attr, 10);
  ASSERT_EQ(int64_attr, 100);
  ASSERT_NEAR(float_attr, 3.14, 1e-6);
  ASSERT_EQ(str_attr, "test");
  ASSERT_EQ(vec_bool_attr.at(0), true);
  ASSERT_EQ(vec_bool_attr.at(1), true);
  ASSERT_EQ(vec_int_attr.at(0), 10);
  ASSERT_EQ(vec_int_attr.at(1), 10);
  ASSERT_EQ(vec_int64_attr.at(0), 100L);
  ASSERT_EQ(vec_int64_attr.at(1), 100L);
  ASSERT_NEAR(vec_float_attr.at(0), 3.14, 1e-6);
  ASSERT_NEAR(vec_float_attr.at(1), 3.14, 1e-6);
  ASSERT_NEAR(vec_double_attr.at(0), 3.1415, 1e-6);
  ASSERT_NEAR(vec_double_attr.at(1), 3.1415, 1e-6);
  ASSERT_EQ(vec_str_attr.at(0), "test_vec");
  ASSERT_EQ(vec_str_attr.at(1), "test_vec");
}

class InferShapeUtilsTestOpMaker : public OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddAttr<bool>("bool", "bool attr of test op");
    AddAttr<int>("int", "int attr of test op");
    AddAttr<int64_t>("int64", "int64 attr of test op");
    AddAttr<float>("float", "float attr of test op");
    AddAttr<std::string>("string", "string attr of test op");
    AddAttr<std::vector<bool>>("vec_bool", "vec_bool attr of test op");
    AddAttr<std::vector<int>>("vec_int", "vec_int attr of test op");
    AddAttr<std::vector<int64_t>>("vec_int64", "vec_int attr of test op");
    AddAttr<std::vector<float>>("vec_float", "vec_int attr of test op");
    AddAttr<std::vector<double>>("vec_double", "vec_int attr of test op");
    AddAttr<std::vector<std::string>>("vec_str", "vec_int attr of test op");
    AddComment("This is test op");
  }
};

class InferShapeUtilsTestOp : public OperatorWithKernel {
 public:
  using OperatorWithKernel::OperatorWithKernel;

  phi::KernelKey GetExpectedKernelType(
      const ExecutionContext& ctx) const override {
    return phi::KernelKey(proto::VarType::FP32, ctx.GetPlace());
  }
};

phi::KernelSignature InferShapeUtilsTestOpArgumentMapping(
    const phi::ArgumentMappingContext& ctx) {
  return phi::KernelSignature("infer_shape_utils_test",
                              {},
                              {"bool",
                               "int",
                               "int64",
                               "float",
                               "string",
                               "vec_bool",
                               "vec_int",
                               "vec_int64",
                               "vec_float",
                               "vec_double",
                               "vec_str"},
                              {});
}

template <typename T, typename Context>
void InferShapeUtilsTestKernel(const Context& dev_ctx,
                               const phi::DenseTensor& x,
                               bool attr1,
                               int attr2,
                               int64_t attr3,
                               float attr4,
                               const std::string& attr5,
                               const std::vector<bool>& attr6,
                               const std::vector<int>& attr7,
                               const std::vector<int64_t>& attr8,
                               const std::vector<float>& attr9,
                               const std::vector<double>& attr10,
                               const std::vector<std::string>& attr11,
                               phi::DenseTensor* out) {
  VLOG(6) << "Come into InferShapeUtilsTestKernel";
}

void TestOutputInferMeta(const phi::MetaTensor& x, phi::MetaTensor* out) {
  ASSERT_EQ(x.dtype(), phi::DataType::FLOAT32);
}

class InferShapeUtilsTestOutputOpMaker : public OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "input of test op");
    AddOutput("Out", "output of test op");
    AddComment("This is test op");
  }
};

class InferShapeUtilsTestOutputOp : public OperatorWithKernel {
 public:
  using OperatorWithKernel::OperatorWithKernel;

  phi::KernelKey GetExpectedKernelType(
      const ExecutionContext& ctx) const override {
    return phi::KernelKey(proto::VarType::FP32, ctx.GetPlace());
  }
};

phi::KernelSignature TestSparseOutputOpArgumentMapping(
    const phi::ArgumentMappingContext& ctx) {
  if (ctx.IsSparseCooTensorOutput("Out")) {
    return phi::KernelSignature(
        "test_sparse_coo_tensor_output", {"X"}, {}, {"Out"});
  }
  return phi::KernelSignature("test_output", {"X"}, {}, {"Out"});
}

template <typename T, typename Context>
void InferShapeUtilsTestOutputKernel(const Context& dev_ctx,
                                     const phi::DenseTensor& x,
                                     phi::SparseCooTensor* out) {
  VLOG(6) << "Come into InferShapeUtilsTestOutputKernel";
}

}  // namespace framework
}  // namespace paddle

DECLARE_INFER_SHAPE_FUNCTOR(infer_shape_utils_test,
                            InferShapeUtilsTestInferShapeFunctor,
                            PD_INFER_META(paddle::framework::TestInferMeta));
REGISTER_OPERATOR(infer_shape_utils_test,
                  paddle::framework::InferShapeUtilsTestOp,
                  paddle::framework::InferShapeUtilsTestOpMaker,
                  InferShapeUtilsTestInferShapeFunctor);

PD_REGISTER_KERNEL(infer_shape_utils_test,
                   CPU,
                   ALL_LAYOUT,
                   paddle::framework::InferShapeUtilsTestKernel,
                   int) {}

DECLARE_INFER_SHAPE_FUNCTOR(
    infer_shape_utils_test_output,
    InferShapeUtilsTestOutputInferShapeFunctor,
    PD_INFER_META(paddle::framework::TestOutputInferMeta));
REGISTER_OPERATOR(infer_shape_utils_test_output,
                  paddle::framework::InferShapeUtilsTestOutputOp,
                  paddle::framework::InferShapeUtilsTestOutputOpMaker,
                  InferShapeUtilsTestOutputInferShapeFunctor);

PD_REGISTER_KERNEL(test_sparse_coo_tensor_output,
                   CPU,
                   ALL_LAYOUT,
                   paddle::framework::InferShapeUtilsTestOutputKernel,
                   int) {}

TEST(InferShapeUtilsTest, ALL) {
  paddle::framework::ProgramDesc prog;
  paddle::framework::proto::BlockDesc proto_block;
  paddle::framework::BlockDesc block_desc(&prog, &proto_block);

  auto* op = block_desc.AppendOp();
  op->SetType("infer_shape_utils_test");

  paddle::framework::Attribute bool_attr(true);
  op->SetAttr("bool", bool_attr);

  paddle::framework::Attribute int_attr(10);
  op->SetAttr("int", int_attr);

  int64_t int64_val = 100;
  paddle::framework::Attribute int64_attr(int64_val);
  op->SetAttr("int64", int64_attr);

  float float_value = 3.14;
  paddle::framework::Attribute float_attr(float_value);
  op->SetAttr("float", float_attr);

  std::string str_value("test");
  paddle::framework::Attribute str_attr(str_value);
  op->SetAttr("string", str_attr);

  std::vector<bool> vec_bool(2, true);
  paddle::framework::Attribute vec_bool_attr = vec_bool;
  op->SetAttr("vec_bool", vec_bool_attr);

  std::vector<int> vec_int(2, 10);
  paddle::framework::Attribute vec_int_attr = vec_int;
  op->SetAttr("vec_int", vec_int_attr);

  std::vector<int64_t> vec_int64(2, 100);
  paddle::framework::Attribute vec_int64_attr = vec_int64;
  op->SetAttr("vec_int64", vec_int64_attr);
  std::cout << "after set vec_int64" << std::endl;

  std::vector<float> vec_float(2, 3.14);
  paddle::framework::Attribute vec_float_attr = vec_float;
  op->SetAttr("vec_float", vec_float_attr);

  std::vector<double> vec_double(2, 3.1415);
  paddle::framework::Attribute vec_double_attr = vec_double;
  op->SetAttr("vec_double", vec_double_attr);

  std::vector<std::string> vec_str(2, "test_vec");
  paddle::framework::Attribute vec_str_attr = vec_str;
  op->SetAttr("vec_str", vec_str_attr);

  phi::OpUtilsMap::Instance().InsertArgumentMappingFn(
      "infer_shape_utils_test",
      paddle::framework::InferShapeUtilsTestOpArgumentMapping);

  op->InferShape(block_desc);
}

TEST(InferShapeUtilsTestOutput, ALL) {
  paddle::framework::ProgramDesc prog;
  paddle::framework::proto::BlockDesc proto_block;
  paddle::framework::BlockDesc block_desc(&prog, &proto_block);

  auto* op = block_desc.AppendOp();
  op->SetType("infer_shape_utils_test_output");

  auto* x = block_desc.Var("x");
  x->SetType(paddle::framework::proto::VarType::DENSE_TENSOR);
  x->SetDataType(paddle::framework::proto::VarType::FP32);
  op->SetInput("X", {"x"});

  auto* out = block_desc.Var("out");
  out->SetType(paddle::framework::proto::VarType::SPARSE_COO);
  op->SetOutput("Out", {"out"});

  phi::OpUtilsMap::Instance().InsertArgumentMappingFn(
      "infer_shape_utils_test_output",
      paddle::framework::TestSparseOutputOpArgumentMapping);

  op->InferShape(block_desc);
}
